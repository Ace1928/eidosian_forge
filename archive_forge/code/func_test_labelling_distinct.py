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
@pytest.mark.parametrize('allow_single_cluster', [True, False])
@pytest.mark.parametrize('epsilon', [0, 0.1])
def test_labelling_distinct(global_random_seed, allow_single_cluster, epsilon):
    """
    Tests that the `_do_labelling` helper function correctly assigns labels.
    """
    n_samples = 48
    X, y = make_blobs(n_samples, random_state=global_random_seed, centers=[[0, 0], [10, 0], [0, 10]])
    est = HDBSCAN().fit(X)
    condensed_tree = _condense_tree(est._single_linkage_tree_, min_cluster_size=est.min_cluster_size)
    clusters = {n_samples + 2, n_samples + 3, n_samples + 4}
    cluster_label_map = {n_samples + 2: 0, n_samples + 3: 1, n_samples + 4: 2}
    labels = _do_labelling(condensed_tree=condensed_tree, clusters=clusters, cluster_label_map=cluster_label_map, allow_single_cluster=allow_single_cluster, cluster_selection_epsilon=epsilon)
    first_with_label = {_y: np.where(y == _y)[0][0] for _y in list(set(y))}
    y_to_labels = {_y: labels[first_with_label[_y]] for _y in list(set(y))}
    aligned_target = np.vectorize(y_to_labels.get)(y)
    assert_array_equal(labels, aligned_target)