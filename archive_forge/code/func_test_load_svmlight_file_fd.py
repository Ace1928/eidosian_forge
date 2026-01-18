import gzip
import os
import shutil
from bz2 import BZ2File
from importlib import resources
from io import BytesIO
from tempfile import NamedTemporaryFile
import numpy as np
import pytest
import scipy.sparse as sp
import sklearn
from sklearn.datasets import dump_svmlight_file, load_svmlight_file, load_svmlight_files
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_load_svmlight_file_fd():
    data_path = resources.files(TEST_DATA_MODULE) / datafile
    data_path = str(data_path)
    X1, y1 = load_svmlight_file(data_path)
    fd = os.open(data_path, os.O_RDONLY)
    try:
        X2, y2 = load_svmlight_file(fd)
        assert_array_almost_equal(X1.data, X2.data)
        assert_array_almost_equal(y1, y2)
    finally:
        os.close(fd)