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
def test_multilabel_y_explicit_zeros(tmp_path, csr_container):
    """
    Ensure that if y contains explicit zeros (i.e. elements of y.data equal to
    0) then those explicit zeros are not encoded.
    """
    save_path = str(tmp_path / 'svm_explicit_zero')
    rng = np.random.RandomState(42)
    X = rng.randn(3, 5).astype(np.float64)
    indptr = np.array([0, 2, 3, 6])
    indices = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([0, 1, 1, 1, 1, 0])
    y = csr_container((data, indices, indptr), shape=(3, 3))
    dump_svmlight_file(X, y, save_path, multilabel=True)
    _, y_load = load_svmlight_file(save_path, multilabel=True)
    y_true = [(2.0,), (2.0,), (0.0, 1.0)]
    assert y_load == y_true