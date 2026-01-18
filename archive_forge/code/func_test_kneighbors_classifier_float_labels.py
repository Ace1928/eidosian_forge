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
def test_kneighbors_classifier_float_labels(global_dtype, n_samples=40, n_features=5, n_test_pts=10, n_neighbors=5, random_state=0):
    rng = np.random.RandomState(random_state)
    X = 2 * rng.rand(n_samples, n_features).astype(global_dtype, copy=False) - 1
    y = ((X ** 2).sum(axis=1) < 0.5).astype(int)
    knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X, y.astype(float))
    epsilon = 1e-05 * (2 * rng.rand(1, n_features) - 1)
    y_pred = knn.predict(X[:n_test_pts] + epsilon)
    assert_array_equal(y_pred, y[:n_test_pts])