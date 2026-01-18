import pickle
import warnings
import numpy as np
import pytest
from scipy.spatial import distance
from sklearn.cluster import DBSCAN, dbscan
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS, LIL_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_dbscan_precomputed_metric_with_initial_rows_zero(csr_container):
    ar = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0], [0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.3], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1], [0.0, 0.0, 0.0, 0.0, 0.3, 0.1, 0.0]])
    matrix = csr_container(ar)
    labels = DBSCAN(eps=0.2, metric='precomputed', min_samples=2).fit(matrix).labels_
    assert_array_equal(labels, [-1, -1, 0, 0, 0, 1, 1])