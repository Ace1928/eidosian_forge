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
def test_load_zero_based_auto():
    data1 = b'-1 1:1 2:2 3:3\n'
    data2 = b'-1 0:0 1:1\n'
    f1 = BytesIO(data1)
    X, y = load_svmlight_file(f1, zero_based='auto')
    assert X.shape == (1, 3)
    f1 = BytesIO(data1)
    f2 = BytesIO(data2)
    X1, y1, X2, y2 = load_svmlight_files([f1, f2], zero_based='auto')
    assert X1.shape == (1, 4)
    assert X2.shape == (1, 4)