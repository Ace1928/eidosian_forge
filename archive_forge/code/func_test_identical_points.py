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
def test_identical_points():
    X = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2]])
    true_labels = np.array([0, 0, 1, 1, 2, 2])
    connectivity = kneighbors_graph(X, n_neighbors=3, include_self=False)
    connectivity = 0.5 * (connectivity + connectivity.T)
    connectivity, n_components = _fix_connectivity(X, connectivity, 'euclidean')
    for linkage in ('single', 'average', 'average', 'ward'):
        clustering = AgglomerativeClustering(n_clusters=3, linkage=linkage, connectivity=connectivity)
        clustering.fit(X)
        assert_almost_equal(normalized_mutual_info_score(clustering.labels_, true_labels), 1)