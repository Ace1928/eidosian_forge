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
@pytest.mark.parametrize('Klass, default_n_init', [(KMeans, 10), (MiniBatchKMeans, 3)])
def test_n_init_auto(Klass, default_n_init):
    est = Klass(n_init='auto', init='k-means++')
    est.fit(X)
    assert est._n_init == 1
    est = Klass(n_init='auto', init='random')
    est.fit(X)
    assert est._n_init == 10 if Klass.__name__ == 'KMeans' else 3