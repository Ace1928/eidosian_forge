import warnings
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
def test_incremental_pca_validation():
    X = np.array([[0, 1, 0], [1, 0, 0]])
    n_samples, n_features = X.shape
    n_components = 4
    with pytest.raises(ValueError, match='n_components={} invalid for n_features={}, need more rows than columns for IncrementalPCA processing'.format(n_components, n_features)):
        IncrementalPCA(n_components, batch_size=10).fit(X)
    n_components = 3
    with pytest.raises(ValueError, match='n_components={} must be less or equal to the batch number of samples {}'.format(n_components, n_samples)):
        IncrementalPCA(n_components=n_components).partial_fit(X)