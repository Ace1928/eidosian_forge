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
def test_nearest_neighbours_works_with_p_less_than_1():
    """Check that NearestNeighbors works with :math:`p \\in (0,1)` when `algorithm`
    is `"auto"` or `"brute"` regardless of the dtype of X.

    Non-regression test for issue #26548
    """
    X = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    neigh = neighbors.NearestNeighbors(n_neighbors=3, algorithm='brute', metric_params={'p': 0.5})
    neigh.fit(X)
    y = neigh.radius_neighbors(X[0].reshape(1, -1), radius=4, return_distance=False)
    assert_allclose(y[0], [0, 1, 2])
    y = neigh.kneighbors(X[0].reshape(1, -1), return_distance=False)
    assert_allclose(y[0], [0, 1, 2])