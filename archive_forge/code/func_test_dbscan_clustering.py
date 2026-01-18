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
def test_dbscan_clustering():
    """
    Tests that HDBSCAN can generate a sufficiently accurate dbscan clustering.
    This test is more of a sanity check than a rigorous evaluation.
    """
    clusterer = HDBSCAN().fit(X)
    labels = clusterer.dbscan_clustering(0.3)
    check_label_quality(labels, threshold=0.92)