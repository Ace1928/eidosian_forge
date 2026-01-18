import warnings
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
def test_incremental_pca_against_pca_random_data():
    rng = np.random.RandomState(1999)
    n_samples = 100
    n_features = 3
    X = rng.randn(n_samples, n_features) + 5 * rng.rand(1, n_features)
    Y_pca = PCA(n_components=3).fit_transform(X)
    Y_ipca = IncrementalPCA(n_components=3, batch_size=25).fit_transform(X)
    assert_almost_equal(np.abs(Y_pca), np.abs(Y_ipca), 1)