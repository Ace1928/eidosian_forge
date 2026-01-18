import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances_argmin, v_measure_score
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
def test_branching_factor(global_random_seed, global_dtype):
    X, y = make_blobs(random_state=global_random_seed)
    X = X.astype(global_dtype, copy=False)
    branching_factor = 9
    brc = Birch(n_clusters=None, branching_factor=branching_factor, threshold=0.01)
    brc.fit(X)
    check_branching_factor(brc.root_, branching_factor)
    brc = Birch(n_clusters=3, branching_factor=branching_factor, threshold=0.01)
    brc.fit(X)
    check_branching_factor(brc.root_, branching_factor)