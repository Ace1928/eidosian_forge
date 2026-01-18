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
def rotten_tomatoes():
    """
    Contains information from kaggle [1], which is made available here under the Open Database License (ODbL) [2].

    Download "rotten_tomatoes" [1] data set.

    Will return two pandas.DataFrame-s, first with train part (rotten_tomatoes.data) and second with test part
    (rotten_tomatoes.test) of the dataset.

    NOTE: This is a preprocessed version of the dataset.

    [1]: https://www.kaggle.com/rpnuser8182/rotten-tomatoes
    [2]: https://opendatacommons.org/licenses/odbl/1-0/index.html
    """
    url = 'https://catboost-opensource.s3.yandex.net/rotten_tomatoes.tar.gz'
    md5 = 'a07fed612805ac9e17ced0d82a96add4'
    dataset_name, train_file, test_file = ('rotten_tomatoes', 'learn.tsv', 'test.tsv')
    return _load_dataset_pd(url, md5, dataset_name, train_file, test_file, sep='\t')