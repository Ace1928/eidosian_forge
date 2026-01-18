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
def test_compute_full_tree():
    rng = np.random.RandomState(0)
    X = rng.randn(10, 2)
    connectivity = kneighbors_graph(X, 5, include_self=False)
    agc = AgglomerativeClustering(n_clusters=2, connectivity=connectivity)
    agc.fit(X)
    n_samples = X.shape[0]
    n_nodes = agc.children_.shape[0]
    assert n_nodes == n_samples - 1
    n_clusters = 101
    X = rng.randn(200, 2)
    connectivity = kneighbors_graph(X, 10, include_self=False)
    agc = AgglomerativeClustering(n_clusters=n_clusters, connectivity=connectivity)
    agc.fit(X)
    n_samples = X.shape[0]
    n_nodes = agc.children_.shape[0]
    assert n_nodes == n_samples - n_clusters