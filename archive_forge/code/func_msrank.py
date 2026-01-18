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
def msrank():
    url = 'https://storage.mds.yandex.net/get-devtools-opensource/233854/msrank.tar.gz'
    md5 = '34fee225d02419adc106581f4eb36f2e'
    dataset_name, train_file, test_file = ('msrank', 'train.tsv', 'test.tsv')
    return _load_dataset_pd(url, md5, dataset_name, train_file, test_file, header=None, sep='\t', cache=True)