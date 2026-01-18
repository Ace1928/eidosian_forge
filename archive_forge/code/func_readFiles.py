import parlai.core.build_data as build_data
import os
import numpy
from parlai.core.build_data import DownloadableFile
def readFiles(dpath, rfnames):
    en_fname, de_fname = rfnames
    with open(os.path.join(dpath, en_fname), 'r') as f:
        en = [l[:-1].replace('##AT##-##AT##', '__AT__') for l in f]
    with open(os.path.join(dpath, de_fname), 'r') as f:
        de = [l[:-1].replace('##AT##-##AT##', '__AT__') for l in f]
    return list(zip(de, en))