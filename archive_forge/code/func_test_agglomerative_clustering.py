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
@pytest.mark.parametrize('lil_container', LIL_CONTAINERS)
def test_agglomerative_clustering(global_random_seed, lil_container):
    rng = np.random.RandomState(global_random_seed)
    mask = np.ones([10, 10], dtype=bool)
    n_samples = 100
    X = rng.randn(n_samples, 50)
    connectivity = grid_to_graph(*mask.shape)
    for linkage in ('ward', 'complete', 'average', 'single'):
        clustering = AgglomerativeClustering(n_clusters=10, connectivity=connectivity, linkage=linkage)
        clustering.fit(X)
        try:
            tempdir = mkdtemp()
            clustering = AgglomerativeClustering(n_clusters=10, connectivity=connectivity, memory=tempdir, linkage=linkage)
            clustering.fit(X)
            labels = clustering.labels_
            assert np.size(np.unique(labels)) == 10
        finally:
            shutil.rmtree(tempdir)
        clustering = AgglomerativeClustering(n_clusters=10, connectivity=connectivity, linkage=linkage)
        clustering.compute_full_tree = False
        clustering.fit(X)
        assert_almost_equal(normalized_mutual_info_score(clustering.labels_, labels), 1)
        clustering.connectivity = None
        clustering.fit(X)
        assert np.size(np.unique(clustering.labels_)) == 10
        clustering = AgglomerativeClustering(n_clusters=10, connectivity=lil_container(connectivity.toarray()[:10, :10]), linkage=linkage)
        with pytest.raises(ValueError):
            clustering.fit(X)
    clustering = AgglomerativeClustering(n_clusters=10, connectivity=connectivity.toarray(), metric='manhattan', linkage='ward')
    with pytest.raises(ValueError):
        clustering.fit(X)
    for metric in PAIRED_DISTANCES.keys():
        clustering = AgglomerativeClustering(n_clusters=10, connectivity=np.ones((n_samples, n_samples)), metric=metric, linkage='complete')
        clustering.fit(X)
        clustering2 = AgglomerativeClustering(n_clusters=10, connectivity=None, metric=metric, linkage='complete')
        clustering2.fit(X)
        assert_almost_equal(normalized_mutual_info_score(clustering2.labels_, clustering.labels_), 1)
    clustering = AgglomerativeClustering(n_clusters=10, connectivity=connectivity, linkage='complete')
    clustering.fit(X)
    X_dist = pairwise_distances(X)
    clustering2 = AgglomerativeClustering(n_clusters=10, connectivity=connectivity, metric='precomputed', linkage='complete')
    clustering2.fit(X_dist)
    assert_array_equal(clustering.labels_, clustering2.labels_)