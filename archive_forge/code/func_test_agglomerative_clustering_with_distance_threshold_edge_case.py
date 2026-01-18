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
@pytest.mark.parametrize(('threshold', 'y_true'), [(0.5, [1, 0]), (1.0, [1, 0]), (1.5, [0, 0])])
def test_agglomerative_clustering_with_distance_threshold_edge_case(linkage, threshold, y_true):
    X = [[0], [1]]
    clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, linkage=linkage)
    y_pred = clusterer.fit_predict(X)
    assert adjusted_rand_score(y_true, y_pred) == 1