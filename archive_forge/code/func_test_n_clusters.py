import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances_argmin, v_measure_score
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
def test_n_clusters(global_random_seed, global_dtype):
    X, y = make_blobs(n_samples=100, centers=10, random_state=global_random_seed)
    X = X.astype(global_dtype, copy=False)
    brc1 = Birch(n_clusters=10)
    brc1.fit(X)
    assert len(brc1.subcluster_centers_) > 10
    assert len(np.unique(brc1.labels_)) == 10
    gc = AgglomerativeClustering(n_clusters=10)
    brc2 = Birch(n_clusters=gc)
    brc2.fit(X)
    assert_array_equal(brc1.subcluster_labels_, brc2.subcluster_labels_)
    assert_array_equal(brc1.labels_, brc2.labels_)
    brc4 = Birch(threshold=10000.0)
    with pytest.warns(ConvergenceWarning):
        brc4.fit(X)