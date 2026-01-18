import warnings
import numpy as np
import pytest
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.cluster._optics import _extend_region, _extract_xi_labels
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.exceptions import DataConversionWarning, EfficiencyWarning
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils import shuffle
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
def test_extract_dbscan(global_dtype):
    rng = np.random.RandomState(0)
    n_points_per_cluster = 20
    C1 = [-5, -2] + 0.2 * rng.randn(n_points_per_cluster, 2)
    C2 = [4, -1] + 0.2 * rng.randn(n_points_per_cluster, 2)
    C3 = [1, 2] + 0.2 * rng.randn(n_points_per_cluster, 2)
    C4 = [-2, 3] + 0.2 * rng.randn(n_points_per_cluster, 2)
    X = np.vstack((C1, C2, C3, C4)).astype(global_dtype, copy=False)
    clust = OPTICS(cluster_method='dbscan', eps=0.5).fit(X)
    assert_array_equal(np.sort(np.unique(clust.labels_)), [0, 1, 2, 3])