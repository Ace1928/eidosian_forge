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
def test_kmeans_plusplus_dataorder(global_random_seed):
    centers_c, _ = kmeans_plusplus(X, n_clusters, random_state=global_random_seed)
    X_fortran = np.asfortranarray(X)
    centers_fortran, _ = kmeans_plusplus(X_fortran, n_clusters, random_state=global_random_seed)
    assert_allclose(centers_c, centers_fortran)