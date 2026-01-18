import warnings
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
def test_incremental_pca_inverse():
    rng = np.random.RandomState(1999)
    n, p = (50, 3)
    X = rng.randn(n, p)
    X[:, 1] *= 1e-05
    X += [5, 4, 3]
    ipca = IncrementalPCA(n_components=2, batch_size=10).fit(X)
    Y = ipca.transform(X)
    Y_inverse = ipca.inverse_transform(Y)
    assert_almost_equal(X, Y_inverse, decimal=3)