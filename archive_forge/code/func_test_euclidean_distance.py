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
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('squared', [True, False])
def test_euclidean_distance(dtype, squared, global_random_seed):
    rng = np.random.RandomState(global_random_seed)
    a_sparse = sp.random(1, 100, density=0.5, format='csr', random_state=rng, dtype=dtype)
    a_dense = a_sparse.toarray().reshape(-1)
    b = rng.randn(100).astype(dtype, copy=False)
    b_squared_norm = (b ** 2).sum()
    expected = ((a_dense - b) ** 2).sum()
    expected = expected if squared else np.sqrt(expected)
    distance_dense_dense = _euclidean_dense_dense_wrapper(a_dense, b, squared)
    distance_sparse_dense = _euclidean_sparse_dense_wrapper(a_sparse.data, a_sparse.indices, b, b_squared_norm, squared)
    rtol = 0.0001 if dtype == np.float32 else 1e-07
    assert_allclose(distance_dense_dense, distance_sparse_dense, rtol=rtol)
    assert_allclose(distance_dense_dense, expected, rtol=rtol)
    assert_allclose(distance_sparse_dense, expected, rtol=rtol)