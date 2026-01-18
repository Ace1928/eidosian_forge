import warnings
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
@pytest.mark.parametrize('sparse_container', CSC_CONTAINERS + CSR_CONTAINERS + LIL_CONTAINERS)
def test_incremental_pca_sparse(sparse_container):
    X = iris.data
    pca = PCA(n_components=2)
    pca.fit_transform(X)
    X_sparse = sparse_container(X)
    batch_size = X_sparse.shape[0] // 3
    ipca = IncrementalPCA(n_components=2, batch_size=batch_size)
    X_transformed = ipca.fit_transform(X_sparse)
    assert X_transformed.shape == (X_sparse.shape[0], 2)
    np.testing.assert_allclose(ipca.explained_variance_ratio_.sum(), pca.explained_variance_ratio_.sum(), rtol=0.001)
    for n_components in [1, 2, X.shape[1]]:
        ipca = IncrementalPCA(n_components, batch_size=batch_size)
        ipca.fit(X_sparse)
        cov = ipca.get_covariance()
        precision = ipca.get_precision()
        np.testing.assert_allclose(np.dot(cov, precision), np.eye(X_sparse.shape[1]), atol=1e-13)
    with pytest.raises(TypeError, match='IncrementalPCA.partial_fit does not support sparse input. Either convert data to dense or use IncrementalPCA.fit to do so in batches.'):
        ipca.partial_fit(X_sparse)