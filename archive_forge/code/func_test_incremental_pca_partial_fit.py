import warnings
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
def test_incremental_pca_partial_fit():
    rng = np.random.RandomState(1999)
    n, p = (50, 3)
    X = rng.randn(n, p)
    X[:, 1] *= 1e-05
    X += [5, 4, 3]
    batch_size = 10
    ipca = IncrementalPCA(n_components=2, batch_size=batch_size).fit(X)
    pipca = IncrementalPCA(n_components=2, batch_size=batch_size)
    batch_itr = np.arange(0, n + 1, batch_size)
    for i, j in zip(batch_itr[:-1], batch_itr[1:]):
        pipca.partial_fit(X[i:j, :])
    assert_almost_equal(ipca.components_, pipca.components_, decimal=3)