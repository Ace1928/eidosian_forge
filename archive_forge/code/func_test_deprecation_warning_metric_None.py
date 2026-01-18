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
@pytest.mark.parametrize('Agglomeration', [AgglomerativeClustering, FeatureAgglomeration])
def test_deprecation_warning_metric_None(Agglomeration):
    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    warn_msg = '`metric=None` is deprecated in version 1.4 and will be removed'
    with pytest.warns(FutureWarning, match=warn_msg):
        Agglomeration(metric=None).fit(X)