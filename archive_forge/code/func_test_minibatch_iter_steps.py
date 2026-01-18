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
def test_minibatch_iter_steps():
    batch_size = 30
    n_samples = X.shape[0]
    km = MiniBatchKMeans(n_clusters=3, batch_size=batch_size, random_state=0).fit(X)
    assert km.n_iter_ == np.ceil(km.n_steps_ * batch_size / n_samples)
    assert isinstance(km.n_iter_, int)
    km = MiniBatchKMeans(n_clusters=3, batch_size=batch_size, random_state=0, tol=0, max_no_improvement=None, max_iter=10).fit(X)
    assert km.n_iter_ == 10
    assert km.n_steps_ == 10 * n_samples // batch_size
    assert isinstance(km.n_steps_, int)