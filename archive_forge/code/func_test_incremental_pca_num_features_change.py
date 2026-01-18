import warnings
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
def test_incremental_pca_num_features_change():
    rng = np.random.RandomState(1999)
    n_samples = 100
    X = rng.randn(n_samples, 20)
    X2 = rng.randn(n_samples, 50)
    ipca = IncrementalPCA(n_components=None)
    ipca.fit(X)
    with pytest.raises(ValueError):
        ipca.partial_fit(X2)