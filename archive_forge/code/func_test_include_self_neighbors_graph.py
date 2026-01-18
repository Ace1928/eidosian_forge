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
def test_include_self_neighbors_graph():
    X = [[2, 3], [4, 5]]
    kng = neighbors.kneighbors_graph(X, 1, include_self=True).toarray()
    kng_not_self = neighbors.kneighbors_graph(X, 1, include_self=False).toarray()
    assert_array_equal(kng, [[1.0, 0.0], [0.0, 1.0]])
    assert_array_equal(kng_not_self, [[0.0, 1.0], [1.0, 0.0]])
    rng = neighbors.radius_neighbors_graph(X, 5.0, include_self=True).toarray()
    rng_not_self = neighbors.radius_neighbors_graph(X, 5.0, include_self=False).toarray()
    assert_array_equal(rng, [[1.0, 1.0], [1.0, 1.0]])
    assert_array_equal(rng_not_self, [[0.0, 1.0], [1.0, 0.0]])