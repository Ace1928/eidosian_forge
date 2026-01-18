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
def test_min_samples_edge_case(global_dtype):
    C1 = [[0, 0], [0, 0.1], [0, -0.1]]
    C2 = [[10, 10], [10, 9], [10, 11]]
    C3 = [[100, 100], [100, 96], [100, 106]]
    X = np.vstack((C1, C2, C3)).astype(global_dtype, copy=False)
    expected_labels = np.r_[[0] * 3, [1] * 3, [2] * 3]
    clust = OPTICS(min_samples=3, max_eps=7, cluster_method='xi', xi=0.04).fit(X)
    assert_array_equal(clust.labels_, expected_labels)
    expected_labels = np.r_[[0] * 3, [1] * 3, [-1] * 3]
    clust = OPTICS(min_samples=3, max_eps=3, cluster_method='xi', xi=0.04).fit(X)
    assert_array_equal(clust.labels_, expected_labels)
    expected_labels = np.r_[[-1] * 9]
    with pytest.warns(UserWarning, match='All reachability values'):
        clust = OPTICS(min_samples=4, max_eps=3, cluster_method='xi', xi=0.04).fit(X)
        assert_array_equal(clust.labels_, expected_labels)