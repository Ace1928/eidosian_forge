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
def amazon():
    url = 'https://storage.mds.yandex.net/get-devtools-opensource/250854/amazon.tar.gz'
    md5 = '8fe3eec12bfd9c4c532b24a181d0aa2c'
    dataset_name, train_file, test_file = ('amazon', 'train.csv', 'test.csv')
    return _load_dataset_pd(url, md5, dataset_name, train_file, test_file)