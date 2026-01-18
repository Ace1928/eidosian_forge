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
def test_labelling_thresholding():
    """
    Tests that the `_do_labelling` helper function correctly thresholds the
    incoming lambda values given various `cluster_selection_epsilon` values.
    """
    n_samples = 5
    MAX_LAMBDA = 1.5
    condensed_tree = np.array([(5, 2, MAX_LAMBDA, 1), (5, 1, 0.1, 1), (5, 0, MAX_LAMBDA, 1), (5, 3, 0.2, 1), (5, 4, 0.3, 1)], dtype=CONDENSED_dtype)
    labels = _do_labelling(condensed_tree=condensed_tree, clusters={n_samples}, cluster_label_map={n_samples: 0, n_samples + 1: 1}, allow_single_cluster=True, cluster_selection_epsilon=1)
    num_noise = condensed_tree['value'] < 1
    assert sum(num_noise) == sum(labels == -1)
    labels = _do_labelling(condensed_tree=condensed_tree, clusters={n_samples}, cluster_label_map={n_samples: 0, n_samples + 1: 1}, allow_single_cluster=True, cluster_selection_epsilon=0)
    num_noise = condensed_tree['value'] < MAX_LAMBDA
    assert sum(num_noise) == sum(labels == -1)