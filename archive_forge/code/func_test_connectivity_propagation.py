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
def test_connectivity_propagation():
    X = np.array([(0.014, 0.12), (0.014, 0.099), (0.014, 0.097), (0.017, 0.153), (0.017, 0.153), (0.018, 0.153), (0.018, 0.153), (0.018, 0.153), (0.018, 0.153), (0.018, 0.153), (0.018, 0.153), (0.018, 0.153), (0.018, 0.152), (0.018, 0.149), (0.018, 0.144)])
    connectivity = kneighbors_graph(X, 10, include_self=False)
    ward = AgglomerativeClustering(n_clusters=4, connectivity=connectivity, linkage='ward')
    ward.fit(X)