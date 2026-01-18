import pickle
import warnings
import numpy as np
import pytest
from scipy.spatial import distance
from sklearn.cluster import DBSCAN, dbscan
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS, LIL_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_dbscan_input_not_modified_precomputed_sparse_nodiag(csr_container):
    """Check that we don't modify in-place the pre-computed sparse matrix.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/27508
    """
    X = np.random.RandomState(0).rand(10, 10)
    np.fill_diagonal(X, 0)
    X = csr_container(X)
    assert all((row != col for row, col in zip(*X.nonzero())))
    X_copy = X.copy()
    dbscan(X, metric='precomputed')
    assert X.nnz == X_copy.nnz
    assert_array_equal(X.toarray(), X_copy.toarray())