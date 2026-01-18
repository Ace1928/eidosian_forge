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
@pytest.mark.parametrize('distribution', ['normal', 'blobs'])
@pytest.mark.parametrize('array_constr', data_containers, ids=data_containers_ids)
@pytest.mark.parametrize('tol', [0.01, 1e-08, 1e-100, 0])
def test_kmeans_elkan_results(distribution, array_constr, tol, global_random_seed):
    rnd = np.random.RandomState(global_random_seed)
    if distribution == 'normal':
        X = rnd.normal(size=(5000, 10))
    else:
        X, _ = make_blobs(random_state=rnd)
    X[X < 0] = 0
    X = array_constr(X)
    km_lloyd = KMeans(n_clusters=5, random_state=global_random_seed, n_init=1, tol=tol)
    km_elkan = KMeans(algorithm='elkan', n_clusters=5, random_state=global_random_seed, n_init=1, tol=tol)
    km_lloyd.fit(X)
    km_elkan.fit(X)
    assert_allclose(km_elkan.cluster_centers_, km_lloyd.cluster_centers_)
    assert_array_equal(km_elkan.labels_, km_lloyd.labels_)
    assert km_elkan.n_iter_ == km_lloyd.n_iter_
    assert km_elkan.inertia_ == pytest.approx(km_lloyd.inertia_, rel=1e-06)