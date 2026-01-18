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
@pytest.mark.parametrize('Estimator', [KMeans, MiniBatchKMeans])
def test_fortran_aligned_data(Estimator, global_random_seed):
    X_fortran = np.asfortranarray(X)
    centers_fortran = np.asfortranarray(centers)
    km_c = Estimator(n_clusters=n_clusters, init=centers, n_init=1, random_state=global_random_seed).fit(X)
    km_f = Estimator(n_clusters=n_clusters, init=centers_fortran, n_init=1, random_state=global_random_seed).fit(X_fortran)
    assert_allclose(km_c.cluster_centers_, km_f.cluster_centers_)
    assert_array_equal(km_c.labels_, km_f.labels_)