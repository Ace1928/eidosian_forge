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
def test_kmeans_empty_cluster_relocated(array_constr):
    X = array_constr([[-1], [1]])
    sample_weight = [1.9, 0.1]
    init = np.array([[-1], [10]])
    km = KMeans(n_clusters=2, init=init, n_init=1)
    km.fit(X, sample_weight=sample_weight)
    assert len(set(km.labels_)) == 2
    assert_allclose(km.cluster_centers_, [[-1], [1]])