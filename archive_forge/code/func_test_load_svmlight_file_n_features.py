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
def test_load_svmlight_file_n_features():
    X, y = _load_svmlight_local_test_file(datafile, n_features=22)
    assert X.indptr.shape[0] == 7
    assert X.shape[0] == 6
    assert X.shape[1] == 22
    for i, j, val in ((0, 2, 2.5), (0, 10, -5.2), (1, 5, 1.0), (1, 12, -3)):
        assert X[i, j] == val
    with pytest.raises(ValueError):
        _load_svmlight_local_test_file(datafile, n_features=20)