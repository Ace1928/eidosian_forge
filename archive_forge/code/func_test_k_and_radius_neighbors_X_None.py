import re
import warnings
from itertools import product
import joblib
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import (
from sklearn.base import clone
from sklearn.exceptions import DataConversionWarning, EfficiencyWarning, NotFittedError
from sklearn.metrics._dist_metrics import (
from sklearn.metrics.pairwise import PAIRWISE_BOOLEAN_FUNCTIONS, pairwise_distances
from sklearn.metrics.tests.test_dist_metrics import BOOL_METRICS
from sklearn.metrics.tests.test_pairwise_distances_reduction import (
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import (
from sklearn.neighbors._base import (
from sklearn.pipeline import make_pipeline
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
from sklearn.utils.validation import check_random_state
@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_k_and_radius_neighbors_X_None(algorithm):
    nn = neighbors.NearestNeighbors(n_neighbors=1, algorithm=algorithm)
    X = [[0], [1]]
    nn.fit(X)
    dist, ind = nn.kneighbors()
    assert_array_equal(dist, [[1], [1]])
    assert_array_equal(ind, [[1], [0]])
    dist, ind = nn.radius_neighbors(None, radius=1.5)
    check_object_arrays(dist, [[1], [1]])
    check_object_arrays(ind, [[1], [0]])
    rng = nn.radius_neighbors_graph(None, radius=1.5)
    kng = nn.kneighbors_graph(None)
    for graph in [rng, kng]:
        assert_array_equal(graph.toarray(), [[0, 1], [1, 0]])
        assert_array_equal(graph.data, [1, 1])
        assert_array_equal(graph.indices, [1, 0])
    X = [[0, 1], [0, 1], [1, 1]]
    nn = neighbors.NearestNeighbors(n_neighbors=2, algorithm=algorithm)
    nn.fit(X)
    assert_array_equal(nn.kneighbors_graph().toarray(), np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0]]))