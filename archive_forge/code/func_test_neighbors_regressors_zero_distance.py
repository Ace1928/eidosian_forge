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
def test_neighbors_regressors_zero_distance():
    X = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.5, 2.5]])
    y = np.array([1.0, 1.5, 2.0, 0.0])
    radius = 0.2
    z = np.array([[1.1, 1.1], [2.0, 2.0]])
    rnn_correct_labels = np.array([1.25, 2.0])
    knn_correct_unif = np.array([1.25, 1.0])
    knn_correct_dist = np.array([1.25, 2.0])
    for algorithm in ALGORITHMS:
        for weights in ['uniform', 'distance']:
            rnn = neighbors.RadiusNeighborsRegressor(radius=radius, weights=weights, algorithm=algorithm)
            rnn.fit(X, y)
            assert_allclose(rnn_correct_labels, rnn.predict(z))
        for weights, corr_labels in zip(['uniform', 'distance'], [knn_correct_unif, knn_correct_dist]):
            knn = neighbors.KNeighborsRegressor(n_neighbors=2, weights=weights, algorithm=algorithm)
            knn.fit(X, y)
            assert_allclose(corr_labels, knn.predict(z))