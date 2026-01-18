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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_min_cluster_size_invalid2(csr_container):
    clust = OPTICS(min_cluster_size=len(X) + 1)
    with pytest.raises(ValueError, match='must be no greater than the '):
        clust.fit(X)
    clust = OPTICS(min_cluster_size=len(X) + 1, metric='euclidean')
    with pytest.raises(ValueError, match='must be no greater than the '):
        clust.fit(csr_container(X))