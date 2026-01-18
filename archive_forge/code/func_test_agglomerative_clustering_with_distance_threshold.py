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
@pytest.mark.parametrize('linkage', ['ward', 'complete', 'average'])
def test_agglomerative_clustering_with_distance_threshold(linkage, global_random_seed):
    rng = np.random.RandomState(global_random_seed)
    mask = np.ones([10, 10], dtype=bool)
    n_samples = 100
    X = rng.randn(n_samples, 50)
    connectivity = grid_to_graph(*mask.shape)
    distance_threshold = 10
    for conn in [None, connectivity]:
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, connectivity=conn, linkage=linkage)
        clustering.fit(X)
        clusters_produced = clustering.labels_
        num_clusters_produced = len(np.unique(clustering.labels_))
        tree_builder = _TREE_BUILDERS[linkage]
        children, n_components, n_leaves, parent, distances = tree_builder(X, connectivity=conn, n_clusters=None, return_distance=True)
        num_clusters_at_threshold = np.count_nonzero(distances >= distance_threshold) + 1
        assert num_clusters_at_threshold == num_clusters_produced
        clusters_at_threshold = _hc_cut(n_clusters=num_clusters_produced, children=children, n_leaves=n_leaves)
        assert np.array_equiv(clusters_produced, clusters_at_threshold)