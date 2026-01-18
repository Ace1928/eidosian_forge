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
@pytest.mark.parametrize('kwargs, X', [({'metric': 'precomputed'}, np.array([[1, np.inf], [np.inf, 1]])), ({'metric': 'precomputed'}, [[1, 2], [2, 1]]), ({}, [[1, 2], [3, 4]])])
def test_hdbscan_usable_inputs(X, kwargs):
    """
    Tests that HDBSCAN works correctly for array-likes and precomputed inputs
    with non-finite points.
    """
    HDBSCAN(min_samples=1, **kwargs).fit(X)