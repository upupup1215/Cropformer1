import torch
from torch.utils.data import DataLoader
from model import SBERT
from trainer import SBERTFineTuner
from dataset import PredictDataset
import numpy as np
import random
import os
import argparse
from trainer import SBERTPredict
import pandas as pd
import time
import numpy as np
from osgeo import gdal
import pandas as pd
import os
os.environ['PROJ_LIB'] = r'D:\anaconda\envs\geoscpt\Lib\site-packages\pyproj\proj_dir\share\proj'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(123)


def writetif(im_data, ref_data, im_width, im_height, path):
    driver = gdal.GetDriverByName("GTiff")
    im_bands = ref_data.RasterCount
    datatype = gdal.GDT_Float32
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
    dataset.SetGeoTransform(ref_data.GetGeoTransform())
    dataset.SetProjection(ref_data.GetProjection())
    dataset.GetRasterBand(1).WriteArray(im_data)
    dataset.FlushCache()
    del dataset

    print("-------------new tif writting completed--------------------")


def SearchFiles(directory, fileType):
    fileList = []
    for root, subDirs, files in os.walk(directory):
        for fileName in files:
            if fileName.endswith(fileType):
                fileList.append(os.path.join(root, fileName))
    return fileList
class Config(object):
#修改时间序列地址
    file_path = r'G:\tsbl\2019_hutubi'
#修改保存tif地址
    path_1 = r'E:\classification_program\cropformer_version3\prediction_tif\tsbl\hutubi'
    pretrain_path = r'E:\classification_program\cropformer_version3\checkpoints\pretrain_huocheng_mid_conv_version3_3l_2res_200_epoch_2019/'
    # pretrain_path = None
    finetune_path = r'E:\classification_program\cropformer_version3\checkpoints\finetune_huocheng_tsbl_mid_conv_version3_3l_21class_cropformer/'
    # isExists = os.path.exists(finetune_path)
    # if not isExists:
    #     os.makedirs(finetune_path)
    # print("创建新的文件夹")
    valid_rate = 0.03
    max_length = 64
    num_features = 4
    epochs = 0
    batch_size = 256
    learning_rate = 2e-5
    dropout = 0.10
    hidden_size = 256
    layers = 3
    attn_heads = 8
    warmup_epochs = 1
    decay_gamma = 0.99

    gradient_clipping = 5.0
    num_classes = 21

if __name__ == "__main__":
    start = time.time()
    config = Config()


    lsgrid = os.listdir(config.file_path)
    for i in range(len(lsgrid)):

        test_file = config.file_path +'/'+str(lsgrid[i])+'/'+'pixel_all.csv'
        # print(test_file)
        print("Loading ",test_file)
        # train_dataset = FinetuneDataset(train_file, config.num_features, config.max_length)
        # valid_dataset = FinetuneDataset(valid_file, config.num_features, config.max_length)
        test_dataset = PredictDataset(test_file, config.num_features, config.max_length)
        print("testing samples: %d" %(test_dataset.TS_num))

        print("Creating Dataloader...")
        # train_data_loader = DataLoader(train_dataset, shuffle=True,
        #                                batch_size=config.batch_size, drop_last=False)
        # valid_data_loader = DataLoader(valid_dataset, shuffle=False,
        #                                batch_size=config.batch_size, drop_last=False)
        test_data_loader = DataLoader(test_dataset, shuffle=False,
                                      batch_size=config.batch_size, drop_last=False)


        print("Initialing SITS-BERT...")
        sbert = SBERT(config.num_features, hidden=config.hidden_size, n_layers=config.layers,
                      attn_heads=config.attn_heads, dropout=config.dropout)

        # print(sbert)
        if config.pretrain_path is not None:
            print("Loading pre-trained model parameters...")
            sbert_path = config.pretrain_path + "checkpoint.bert.pth"
            sbert.load_state_dict(torch.load(sbert_path))

        print("Creating prediction...")
        trainer = SBERTPredict(sbert, config.num_classes,)

        print("prediction...")

        trainer.load(config.finetune_path)
        pred_= trainer.test(test_data_loader,config.finetune_path,config.epochs,config.batch_size)

            # print(sublist, end=' ')
        pred = pd.DataFrame(data=pred_)
        # print()
        # data = pred.fillna('Nan')
        data1 = pred.stack().to_frame(0).T

        data1.columns = data1.columns.map('{0[1]}_{0[0]}'.format)
        data1 = data1.T


        # data1.to_csv(r'E:\classification_program\cropformer_version3\prediction_csv/pred.csv',index=None,header=None)
        ####制图
        tif_file = config.file_path + '/' + str(lsgrid[i])
        # tiffile = os.listdir(tif_file)
        fileList = SearchFiles(tif_file, '.tif')
        ref_data_1 = gdal.Open(fileList[0])
        # print(type(ref_data_1))

        # shp_file_path = r"E:\classification_program\cropformer_version3\prediction_csv/pred.csv"

        df = data1

        labels = df.iloc[:, 0]
        labels = np.array(labels)
        label = labels.reshape(625, -1)

        isExists = os.path.exists(config.path_1)
        if not isExists:
            os.makedirs(config.path_1)
        path_2 = config.path_1+'/'+'{}.tif'.format(lsgrid[i])
        writetif(label, ref_data_1, 625, 625, path_2)

    end = time.time()
    print(end-start,'s')


