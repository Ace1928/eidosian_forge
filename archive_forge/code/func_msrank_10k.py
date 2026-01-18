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
def msrank_10k():
    url = 'https://storage.mds.yandex.net/get-devtools-opensource/250854/msrank_10k.tar.gz'
    md5 = '79c5b67397289c4c8b367c1f34629eae'
    dataset_name, train_file, test_file = ('msrank_10k', 'train.csv', 'test.csv')
    return _load_dataset_pd(url, md5, dataset_name, train_file, test_file, header=None)