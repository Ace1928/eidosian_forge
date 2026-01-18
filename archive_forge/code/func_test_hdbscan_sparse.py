import numpy as np
import pytest
from scipy import stats
from scipy.spatial import distance
from sklearn.cluster import HDBSCAN
from sklearn.cluster._hdbscan._tree import (
from sklearn.cluster._hdbscan.hdbscan import _OUTLIER_ENCODING
from sklearn.datasets import make_blobs
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics.pairwise import _VALID_METRICS, euclidean_distances
from sklearn.neighbors import BallTree, KDTree
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_hdbscan_sparse(csr_container):
    """
    Tests that HDBSCAN works correctly when passing sparse feature data.
    Evaluates correctness by comparing against the same data passed as a dense
    array.
    """
    dense_labels = HDBSCAN().fit(X).labels_
    check_label_quality(dense_labels)
    _X_sparse = csr_container(X)
    X_sparse = _X_sparse.copy()
    sparse_labels = HDBSCAN().fit(X_sparse).labels_
    assert_array_equal(dense_labels, sparse_labels)
    for outlier_val, outlier_type in ((np.inf, 'infinite'), (np.nan, 'missing')):
        X_dense = X.copy()
        X_dense[0, 0] = outlier_val
        dense_labels = HDBSCAN().fit(X_dense).labels_
        check_label_quality(dense_labels)
        assert dense_labels[0] == _OUTLIER_ENCODING[outlier_type]['label']
        X_sparse = _X_sparse.copy()
        X_sparse[0, 0] = outlier_val
        sparse_labels = HDBSCAN().fit(X_sparse).labels_
        assert_array_equal(dense_labels, sparse_labels)
    msg = 'Sparse data matrices only support algorithm `brute`.'
    with pytest.raises(ValueError, match=msg):
        HDBSCAN(metric='euclidean', algorithm='ball_tree').fit(X_sparse)