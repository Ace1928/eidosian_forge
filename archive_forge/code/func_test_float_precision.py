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
@pytest.mark.parametrize('input_data', [X] + X_as_any_csr, ids=data_containers_ids)
@pytest.mark.parametrize('Estimator', [KMeans, MiniBatchKMeans])
def test_float_precision(Estimator, input_data, global_random_seed):
    km = Estimator(n_init=1, random_state=global_random_seed)
    inertia = {}
    Xt = {}
    centers = {}
    labels = {}
    for dtype in [np.float64, np.float32]:
        X = input_data.astype(dtype, copy=False)
        km.fit(X)
        inertia[dtype] = km.inertia_
        Xt[dtype] = km.transform(X)
        centers[dtype] = km.cluster_centers_
        labels[dtype] = km.labels_
        assert km.cluster_centers_.dtype == dtype
        if Estimator is MiniBatchKMeans:
            km.partial_fit(X[0:3])
            assert km.cluster_centers_.dtype == dtype
    assert_allclose(inertia[np.float32], inertia[np.float64], rtol=0.0001)
    assert_allclose(Xt[np.float32], Xt[np.float64], atol=Xt[np.float64].max() * 0.0001)
    assert_allclose(centers[np.float32], centers[np.float64], atol=centers[np.float64].max() * 0.0001)
    assert_array_equal(labels[np.float32], labels[np.float64])