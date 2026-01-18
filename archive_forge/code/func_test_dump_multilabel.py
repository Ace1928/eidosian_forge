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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_dump_multilabel(csr_container):
    X = [[1, 0, 3, 0, 5], [0, 0, 0, 0, 0], [0, 5, 0, 1, 0]]
    y_dense = [[0, 1, 0], [1, 0, 1], [1, 1, 0]]
    y_sparse = csr_container(y_dense)
    for y in [y_dense, y_sparse]:
        f = BytesIO()
        dump_svmlight_file(X, y, f, multilabel=True)
        f.seek(0)
        assert f.readline() == b'1 0:1 2:3 4:5\n'
        assert f.readline() == b'0,2 \n'
        assert f.readline() == b'0,1 1:5 3:1\n'