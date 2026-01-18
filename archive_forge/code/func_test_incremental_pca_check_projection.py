import warnings
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
def test_incremental_pca_check_projection():
    rng = np.random.RandomState(1999)
    n, p = (100, 3)
    X = rng.randn(n, p) * 0.1
    X[:10] += np.array([3, 4, 5])
    Xt = 0.1 * rng.randn(1, p) + np.array([3, 4, 5])
    Yt = IncrementalPCA(n_components=2).fit(X).transform(Xt)
    Yt /= np.sqrt((Yt ** 2).sum())
    assert_almost_equal(np.abs(Yt[0][0]), 1.0, 1)