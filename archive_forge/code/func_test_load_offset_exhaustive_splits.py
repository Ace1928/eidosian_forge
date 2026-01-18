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
def test_load_offset_exhaustive_splits(csr_container):
    rng = np.random.RandomState(0)
    X = np.array([[0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 0, 6], [1, 2, 3, 4, 0, 6], [0, 0, 0, 0, 0, 0], [1, 0, 3, 0, 0, 0], [0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0]])
    X = csr_container(X)
    n_samples, n_features = X.shape
    y = rng.randint(low=0, high=2, size=n_samples)
    query_id = np.arange(n_samples) // 2
    f = BytesIO()
    dump_svmlight_file(X, y, f, query_id=query_id)
    f.seek(0)
    size = len(f.getvalue())
    for mark in range(size):
        f.seek(0)
        X_0, y_0, q_0 = load_svmlight_file(f, n_features=n_features, query_id=True, offset=0, length=mark)
        X_1, y_1, q_1 = load_svmlight_file(f, n_features=n_features, query_id=True, offset=mark, length=-1)
        q_concat = np.concatenate([q_0, q_1])
        y_concat = np.concatenate([y_0, y_1])
        X_concat = sp.vstack([X_0, X_1])
        assert_array_almost_equal(y, y_concat)
        assert_array_equal(query_id, q_concat)
        assert_array_almost_equal(X.toarray(), X_concat.toarray())