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
def test_dbscan_metric_params():
    eps = 0.8
    min_samples = 10
    p = 1
    with warnings.catch_warnings(record=True) as warns:
        db = DBSCAN(metric='minkowski', metric_params={'p': p}, eps=eps, p=None, min_samples=min_samples, algorithm='ball_tree').fit(X)
    assert not warns, warns[0].message
    core_sample_1, labels_1 = (db.core_sample_indices_, db.labels_)
    db = DBSCAN(metric='minkowski', eps=eps, min_samples=min_samples, algorithm='ball_tree', p=p).fit(X)
    core_sample_2, labels_2 = (db.core_sample_indices_, db.labels_)
    assert_array_equal(core_sample_1, core_sample_2)
    assert_array_equal(labels_1, labels_2)
    db = DBSCAN(metric='manhattan', eps=eps, min_samples=min_samples, algorithm='ball_tree').fit(X)
    core_sample_3, labels_3 = (db.core_sample_indices_, db.labels_)
    assert_array_equal(core_sample_1, core_sample_3)
    assert_array_equal(labels_1, labels_3)
    with pytest.warns(SyntaxWarning, match='Parameter p is found in metric_params. The corresponding parameter from __init__ is ignored.'):
        db = DBSCAN(metric='minkowski', metric_params={'p': p}, eps=eps, p=p + 1, min_samples=min_samples, algorithm='ball_tree').fit(X)
        core_sample_4, labels_4 = (db.core_sample_indices_, db.labels_)
    assert_array_equal(core_sample_1, core_sample_4)
    assert_array_equal(labels_1, labels_4)