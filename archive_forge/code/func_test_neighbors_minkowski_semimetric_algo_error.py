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
@pytest.mark.parametrize('Estimator', [neighbors.KNeighborsClassifier, neighbors.RadiusNeighborsClassifier, neighbors.KNeighborsRegressor, neighbors.RadiusNeighborsRegressor])
@pytest.mark.parametrize('n_features', [2, 100])
@pytest.mark.parametrize('algorithm', ['kd_tree', 'ball_tree'])
def test_neighbors_minkowski_semimetric_algo_error(Estimator, n_features, algorithm):
    """Check that we raise a proper error if `algorithm!='brute'` and `p<1`."""
    X = rng.random_sample((10, 2))
    y = np.ones(10)
    model = Estimator(algorithm=algorithm, p=0.1)
    msg = f'algorithm="{algorithm}" does not support 0 < p < 1 for the Minkowski metric. To resolve this problem either set p >= 1 or algorithm="brute".'
    with pytest.raises(ValueError, match=msg):
        model.fit(X, y)