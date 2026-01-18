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
@pytest.mark.parametrize('cut_distance', (0.1, 0.5, 1))
def test_dbscan_clustering_outlier_data(cut_distance):
    """
    Tests if np.inf and np.nan data are each treated as special outliers.
    """
    missing_label = _OUTLIER_ENCODING['missing']['label']
    infinite_label = _OUTLIER_ENCODING['infinite']['label']
    X_outlier = X.copy()
    X_outlier[0] = [np.inf, 1]
    X_outlier[2] = [1, np.nan]
    X_outlier[5] = [np.inf, np.nan]
    model = HDBSCAN().fit(X_outlier)
    labels = model.dbscan_clustering(cut_distance=cut_distance)
    missing_labels_idx = np.flatnonzero(labels == missing_label)
    assert_array_equal(missing_labels_idx, [2, 5])
    infinite_labels_idx = np.flatnonzero(labels == infinite_label)
    assert_array_equal(infinite_labels_idx, [0])
    clean_idx = list(set(range(200)) - set(missing_labels_idx + infinite_labels_idx))
    clean_model = HDBSCAN().fit(X_outlier[clean_idx])
    clean_labels = clean_model.dbscan_clustering(cut_distance=cut_distance)
    assert_array_equal(clean_labels, labels[clean_idx])