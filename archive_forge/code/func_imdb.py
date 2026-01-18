import hashlib
import logging
import numpy as np
import os
import pandas as pd
import tarfile
import tempfile
import six
import shutil
from .core import PATH_TYPES, fspath
def imdb():
    url = 'https://catboost-opensource.s3.yandex.net/imdb.tar.gz'
    md5 = '0fd62578d631ac3d71a71c3e6ced6f8b'
    dataset_name, train_file, test_file = ('imdb', 'learn.tsv', 'test.tsv')
    return _load_dataset_pd(url, md5, dataset_name, train_file, test_file, sep='\t')