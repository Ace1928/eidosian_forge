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
@pytest.mark.parametrize('eps', [0.1, 0.3, 0.5])
@pytest.mark.parametrize('min_samples', [3, 10, 20])
@pytest.mark.parametrize('csr_container, metric', [(None, 'minkowski'), (None, 'euclidean')] + [(container, 'euclidean') for container in CSR_CONTAINERS])
def test_dbscan_optics_parity(eps, min_samples, metric, global_dtype, csr_container):
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=150, centers=centers, cluster_std=0.4, random_state=0)
    X = csr_container(X) if csr_container is not None else X
    X = X.astype(global_dtype, copy=False)
    op = OPTICS(min_samples=min_samples, cluster_method='dbscan', eps=eps, metric=metric).fit(X)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    contingency = contingency_matrix(db.labels_, op.labels_)
    agree = min(np.sum(np.max(contingency, axis=0)), np.sum(np.max(contingency, axis=1)))
    disagree = X.shape[0] - agree
    percent_mismatch = np.round((disagree - 1) / X.shape[0], 2)
    assert percent_mismatch <= 0.05