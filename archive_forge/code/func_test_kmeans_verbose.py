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
@pytest.mark.parametrize('algorithm', ['lloyd', 'elkan'])
@pytest.mark.parametrize('tol', [0.01, 0])
def test_kmeans_verbose(algorithm, tol, capsys):
    X = np.random.RandomState(0).normal(size=(5000, 10))
    KMeans(algorithm=algorithm, n_clusters=n_clusters, random_state=42, init='random', n_init=1, tol=tol, verbose=1).fit(X)
    captured = capsys.readouterr()
    assert re.search('Initialization complete', captured.out)
    assert re.search('Iteration [0-9]+, inertia', captured.out)
    if tol == 0:
        assert re.search('strict convergence', captured.out)
    else:
        assert re.search('center shift .* within tolerance', captured.out)