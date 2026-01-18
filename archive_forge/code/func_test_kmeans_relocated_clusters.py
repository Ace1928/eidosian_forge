import re
import sys
from io import StringIO
import numpy as np
import pytest
from scipy import sparse as sp
from sklearn.base import clone
from sklearn.cluster import KMeans, MiniBatchKMeans, k_means, kmeans_plusplus
from sklearn.cluster._k_means_common import (
from sklearn.cluster._kmeans import _labels_inertia, _mini_batch_step
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils._testing import (
from sklearn.utils.extmath import row_norms
from sklearn.utils.fixes import CSR_CONTAINERS, threadpool_limits
@pytest.mark.parametrize('array_constr', data_containers, ids=data_containers_ids)
@pytest.mark.parametrize('algo', ['lloyd', 'elkan'])
def test_kmeans_relocated_clusters(array_constr, algo):
    X = array_constr([[0, 0], [0.5, 0], [0.5, 1], [1, 1]])
    init_centers = np.array([[0.5, 0.5], [3, 3]])
    kmeans = KMeans(n_clusters=2, n_init=1, init=init_centers, algorithm=algo)
    kmeans.fit(X)
    expected_n_iter = 3
    expected_inertia = 0.25
    assert_allclose(kmeans.inertia_, expected_inertia)
    assert kmeans.n_iter_ == expected_n_iter
    try:
        expected_labels = [0, 0, 1, 1]
        expected_centers = [[0.25, 0], [0.75, 1]]
        assert_array_equal(kmeans.labels_, expected_labels)
        assert_allclose(kmeans.cluster_centers_, expected_centers)
    except AssertionError:
        expected_labels = [1, 1, 0, 0]
        expected_centers = [[0.75, 1.0], [0.25, 0.0]]
        assert_array_equal(kmeans.labels_, expected_labels)
        assert_allclose(kmeans.cluster_centers_, expected_centers)