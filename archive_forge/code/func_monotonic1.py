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
def monotonic1():
    """
    Dataset with monotonic constraints.
    Can be used for poisson regression.
    Has several numerical and several categorical features.
    The first column contains target values. Columns with names Cat* contain categorical features.
    Columns with names Num* contain numerical features.

    Dataset also contains several numerical features, for which monotonic constraints must hold.
    For features in columns named MonotonicNeg*, if feature value decreases, then prediction value must not decrease.
    Thus, if there are two samples x1, x2 with all features being equal except
    for a monotonic negative feature M, such that x1[M] > x2[M], then the following inequality must
    hold for predictions: f(x1) <= f(x2)
    """
    url = 'https://storage.mds.yandex.net/get-devtools-opensource/479623/monotonic1.tar.gz'
    md5 = '1b9d8e15bc3fd6f1498e652e7fc4f4ca'
    dataset_name, train_file, test_file = ('monotonic1', 'train.tsv', 'test.tsv')
    return _load_dataset_pd(url, md5, dataset_name, train_file, test_file, sep='\t', cache=True)