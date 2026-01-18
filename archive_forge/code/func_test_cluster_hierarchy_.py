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
def test_cluster_hierarchy_(global_dtype):
    rng = np.random.RandomState(0)
    n_points_per_cluster = 100
    C1 = [0, 0] + 2 * rng.randn(n_points_per_cluster, 2).astype(global_dtype, copy=False)
    C2 = [0, 0] + 50 * rng.randn(n_points_per_cluster, 2).astype(global_dtype, copy=False)
    X = np.vstack((C1, C2))
    X = shuffle(X, random_state=0)
    clusters = OPTICS(min_samples=20, xi=0.1).fit(X).cluster_hierarchy_
    assert clusters.shape == (2, 2)
    diff = np.sum(clusters - np.array([[0, 99], [0, 199]]))
    assert diff / len(X) < 0.05