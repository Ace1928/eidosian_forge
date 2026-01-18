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
def test_dbscan_callable():
    eps = 0.8
    min_samples = 10
    metric = distance.euclidean
    core_samples, labels = dbscan(X, metric=metric, eps=eps, min_samples=min_samples, algorithm='ball_tree')
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    assert n_clusters_1 == n_clusters
    db = DBSCAN(metric=metric, eps=eps, min_samples=min_samples, algorithm='ball_tree')
    labels = db.fit(X).labels_
    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    assert n_clusters_2 == n_clusters