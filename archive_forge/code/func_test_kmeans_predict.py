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
@pytest.mark.parametrize('Estimator, algorithm', [(KMeans, 'lloyd'), (KMeans, 'elkan'), (MiniBatchKMeans, None)])
@pytest.mark.parametrize('max_iter', [2, 100])
def test_kmeans_predict(Estimator, algorithm, array_constr, max_iter, global_dtype, global_random_seed):
    X, _ = make_blobs(n_samples=200, n_features=10, centers=10, random_state=global_random_seed)
    X = array_constr(X, dtype=global_dtype)
    km = Estimator(n_clusters=10, init='random', n_init=10, max_iter=max_iter, random_state=global_random_seed)
    if algorithm is not None:
        km.set_params(algorithm=algorithm)
    km.fit(X)
    labels = km.labels_
    pred = km.predict(X)
    assert_array_equal(pred, labels)
    pred = km.fit_predict(X)
    assert_array_equal(pred, labels)
    pred = km.predict(km.cluster_centers_)
    assert_array_equal(pred, np.arange(10))