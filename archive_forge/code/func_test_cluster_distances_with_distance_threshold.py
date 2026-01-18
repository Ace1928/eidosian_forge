import itertools
import shutil
from functools import partial
from tempfile import mkdtemp
import numpy as np
import pytest
from scipy.cluster import hierarchy
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration, ward_tree
from sklearn.cluster._agglomerative import (
from sklearn.cluster._hierarchical_fast import (
from sklearn.datasets import make_circles, make_moons
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.metrics import DistanceMetric
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import (
from sklearn.metrics.tests.test_dist_metrics import METRICS_DEFAULT_PARAMS
from sklearn.neighbors import kneighbors_graph
from sklearn.utils._fast_dict import IntFloatDict
from sklearn.utils._testing import (
from sklearn.utils.fixes import LIL_CONTAINERS
def test_cluster_distances_with_distance_threshold(global_random_seed):
    rng = np.random.RandomState(global_random_seed)
    n_samples = 100
    X = rng.randint(-10, 10, size=(n_samples, 3))
    distance_threshold = 4
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, linkage='single').fit(X)
    labels = clustering.labels_
    D = pairwise_distances(X, metric='minkowski', p=2)
    np.fill_diagonal(D, np.inf)
    for label in np.unique(labels):
        in_cluster_mask = labels == label
        max_in_cluster_distance = D[in_cluster_mask][:, in_cluster_mask].min(axis=0).max()
        min_out_cluster_distance = D[in_cluster_mask][:, ~in_cluster_mask].min(axis=0).min()
        if in_cluster_mask.sum() > 1:
            assert max_in_cluster_distance < distance_threshold
        assert min_out_cluster_distance >= distance_threshold