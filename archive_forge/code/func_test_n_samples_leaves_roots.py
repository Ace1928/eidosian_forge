import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances_argmin, v_measure_score
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
def test_n_samples_leaves_roots(global_random_seed, global_dtype):
    X, y = make_blobs(n_samples=10, random_state=global_random_seed)
    X = X.astype(global_dtype, copy=False)
    brc = Birch()
    brc.fit(X)
    n_samples_root = sum([sc.n_samples_ for sc in brc.root_.subclusters_])
    n_samples_leaves = sum([sc.n_samples_ for leaf in brc._get_leaves() for sc in leaf.subclusters_])
    assert n_samples_leaves == X.shape[0]
    assert n_samples_root == X.shape[0]