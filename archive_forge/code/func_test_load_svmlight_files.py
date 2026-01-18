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
def test_load_svmlight_files():
    data_path = _svmlight_local_test_file_path(datafile)
    X_train, y_train, X_test, y_test = load_svmlight_files([str(data_path)] * 2, dtype=np.float32)
    assert_array_equal(X_train.toarray(), X_test.toarray())
    assert_array_almost_equal(y_train, y_test)
    assert X_train.dtype == np.float32
    assert X_test.dtype == np.float32
    X1, y1, X2, y2, X3, y3 = load_svmlight_files([str(data_path)] * 3, dtype=np.float64)
    assert X1.dtype == X2.dtype
    assert X2.dtype == X3.dtype
    assert X3.dtype == np.float64