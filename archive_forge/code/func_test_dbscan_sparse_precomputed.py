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
@pytest.mark.parametrize('include_self', [False, True])
def test_dbscan_sparse_precomputed(include_self):
    D = pairwise_distances(X)
    nn = NearestNeighbors(radius=0.9).fit(X)
    X_ = X if include_self else None
    D_sparse = nn.radius_neighbors_graph(X=X_, mode='distance')
    assert D_sparse.nnz < D.shape[0] * (D.shape[0] - 1)
    core_sparse, labels_sparse = dbscan(D_sparse, eps=0.8, min_samples=10, metric='precomputed')
    core_dense, labels_dense = dbscan(D, eps=0.8, min_samples=10, metric='precomputed')
    assert_array_equal(core_dense, core_sparse)
    assert_array_equal(labels_dense, labels_sparse)