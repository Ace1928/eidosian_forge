import warnings
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
def test_incremental_pca_batch_rank():
    rng = np.random.RandomState(1999)
    n_samples = 100
    n_features = 20
    X = rng.randn(n_samples, n_features)
    all_components = []
    batch_sizes = np.arange(20, 90, 3)
    for batch_size in batch_sizes:
        ipca = IncrementalPCA(n_components=20, batch_size=batch_size).fit(X)
        all_components.append(ipca.components_)
    for components_i, components_j in zip(all_components[:-1], all_components[1:]):
        assert_allclose_dense_sparse(components_i, components_j)