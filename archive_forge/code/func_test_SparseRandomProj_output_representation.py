import functools
import warnings
from typing import Any, List
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.exceptions import DataDimensionalityWarning, NotFittedError
from sklearn.metrics import euclidean_distances
from sklearn.random_projection import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS
@pytest.mark.parametrize('coo_container', COO_CONTAINERS)
def test_SparseRandomProj_output_representation(coo_container):
    dense_data = make_sparse_random_data(coo_container, n_samples, n_features, n_nonzeros, random_state=0, sparse_format=None)
    sparse_data = make_sparse_random_data(coo_container, n_samples, n_features, n_nonzeros, random_state=0, sparse_format='csr')
    for SparseRandomProj in all_SparseRandomProjection:
        rp = SparseRandomProj(n_components=10, dense_output=True, random_state=0)
        rp.fit(dense_data)
        assert isinstance(rp.transform(dense_data), np.ndarray)
        assert isinstance(rp.transform(sparse_data), np.ndarray)
        rp = SparseRandomProj(n_components=10, dense_output=False, random_state=0)
        rp = rp.fit(dense_data)
        assert isinstance(rp.transform(dense_data), np.ndarray)
        assert sp.issparse(rp.transform(sparse_data))