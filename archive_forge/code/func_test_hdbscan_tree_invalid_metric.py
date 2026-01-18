import numpy as np
import pytest
from scipy import stats
from scipy.spatial import distance
from sklearn.cluster import HDBSCAN
from sklearn.cluster._hdbscan._tree import (
from sklearn.cluster._hdbscan.hdbscan import _OUTLIER_ENCODING
from sklearn.datasets import make_blobs
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics.pairwise import _VALID_METRICS, euclidean_distances
from sklearn.neighbors import BallTree, KDTree
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_hdbscan_tree_invalid_metric():
    """
    Tests that HDBSCAN correctly raises an error for invalid metric choices.
    """
    metric_callable = lambda x: x
    msg = '.* is not a valid metric for a .*-based algorithm\\. Please select a different metric\\.'
    with pytest.raises(ValueError, match=msg):
        HDBSCAN(algorithm='kd_tree', metric=metric_callable).fit(X)
    with pytest.raises(ValueError, match=msg):
        HDBSCAN(algorithm='ball_tree', metric=metric_callable).fit(X)
    metrics_not_kd = list(set(BallTree.valid_metrics) - set(KDTree.valid_metrics))
    if len(metrics_not_kd) > 0:
        with pytest.raises(ValueError, match=msg):
            HDBSCAN(algorithm='kd_tree', metric=metrics_not_kd[0]).fit(X)