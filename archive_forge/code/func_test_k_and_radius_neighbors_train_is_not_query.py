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
def test_k_and_radius_neighbors_train_is_not_query():
    for algorithm in ALGORITHMS:
        nn = neighbors.NearestNeighbors(n_neighbors=1, algorithm=algorithm)
        X = [[0], [1]]
        nn.fit(X)
        test_data = [[2], [1]]
        dist, ind = nn.kneighbors(test_data)
        assert_array_equal(dist, [[1], [0]])
        assert_array_equal(ind, [[1], [1]])
        dist, ind = nn.radius_neighbors([[2], [1]], radius=1.5)
        check_object_arrays(dist, [[1], [1, 0]])
        check_object_arrays(ind, [[1], [0, 1]])
        assert_array_equal(nn.kneighbors_graph(test_data).toarray(), [[0.0, 1.0], [0.0, 1.0]])
        assert_array_equal(nn.kneighbors_graph([[2], [1]], mode='distance').toarray(), np.array([[0.0, 1.0], [0.0, 0.0]]))
        rng = nn.radius_neighbors_graph([[2], [1]], radius=1.5)
        assert_array_equal(rng.toarray(), [[0, 1], [1, 1]])