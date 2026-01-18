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
@pytest.mark.parametrize('init, expected_n_init', [('k-means++', 1), ('random', 'default'), (lambda X, n_clusters, random_state: random_state.uniform(size=(n_clusters, X.shape[1])), 'default'), ('array-like', 1)])
@pytest.mark.parametrize('Estimator', [KMeans, MiniBatchKMeans])
def test_kmeans_init_auto_with_initial_centroids(Estimator, init, expected_n_init):
    """Check that `n_init="auto"` chooses the right number of initializations.
    Non-regression test for #26657:
    https://github.com/scikit-learn/scikit-learn/pull/26657
    """
    n_sample, n_features, n_clusters = (100, 10, 5)
    X = np.random.randn(n_sample, n_features)
    if init == 'array-like':
        init = np.random.randn(n_clusters, n_features)
    if expected_n_init == 'default':
        expected_n_init = 3 if Estimator is MiniBatchKMeans else 10
    kmeans = Estimator(n_clusters=n_clusters, init=init, n_init='auto').fit(X)
    assert kmeans._n_init == expected_n_init