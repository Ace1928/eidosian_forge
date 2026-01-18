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
@pytest.mark.parametrize('metric_param_grid', METRICS_DEFAULT_PARAMS)
def test_mst_linkage_core_memory_mapped(metric_param_grid):
    """The MST-LINKAGE-CORE algorithm must work on mem-mapped dataset.

    Non-regression test for issue #19875.
    """
    rng = np.random.RandomState(seed=1)
    X = rng.normal(size=(20, 4))
    Xmm = create_memmap_backed_data(X)
    metric, param_grid = metric_param_grid
    keys = param_grid.keys()
    for vals in itertools.product(*param_grid.values()):
        kwargs = dict(zip(keys, vals))
        distance_metric = DistanceMetric.get_metric(metric, **kwargs)
        mst = mst_linkage_core(X, distance_metric)
        mst_mm = mst_linkage_core(Xmm, distance_metric)
        np.testing.assert_equal(mst, mst_mm)