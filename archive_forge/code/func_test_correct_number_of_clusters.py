import warnings
import numpy as np
import pytest
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.cluster._optics import _extend_region, _extract_xi_labels
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.exceptions import DataConversionWarning, EfficiencyWarning
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils import shuffle
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('csr_container, metric', [(None, 'minkowski')] + [(container, 'euclidean') for container in CSR_CONTAINERS])
def test_correct_number_of_clusters(metric, csr_container):
    n_clusters = 3
    X = generate_clustered_data(n_clusters=n_clusters)
    clust = OPTICS(max_eps=5.0 * 6.0, min_samples=4, xi=0.1, metric=metric)
    clust.fit(csr_container(X) if csr_container is not None else X)
    n_clusters_1 = len(set(clust.labels_)) - int(-1 in clust.labels_)
    assert n_clusters_1 == n_clusters
    assert clust.labels_.shape == (len(X),)
    assert clust.labels_.dtype.kind == 'i'
    assert clust.reachability_.shape == (len(X),)
    assert clust.reachability_.dtype.kind == 'f'
    assert clust.core_distances_.shape == (len(X),)
    assert clust.core_distances_.dtype.kind == 'f'
    assert clust.ordering_.shape == (len(X),)
    assert clust.ordering_.dtype.kind == 'i'
    assert set(clust.ordering_) == set(range(len(X)))