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
def test_kneighbors_classifier_predict_proba(global_dtype):
    X = np.array([[0, 2, 0], [0, 2, 1], [2, 0, 0], [2, 2, 0], [0, 0, 2], [0, 0, 1]]).astype(global_dtype, copy=False)
    y = np.array([4, 4, 5, 5, 1, 1])
    cls = neighbors.KNeighborsClassifier(n_neighbors=3, p=1)
    cls.fit(X, y)
    y_prob = cls.predict_proba(X)
    real_prob = np.array([[0, 2, 1], [1, 2, 0], [1, 0, 2], [0, 1, 2], [2, 1, 0], [2, 1, 0]]) / 3.0
    assert_array_equal(real_prob, y_prob)
    cls.fit(X, y.astype(str))
    y_prob = cls.predict_proba(X)
    assert_array_equal(real_prob, y_prob)
    cls = neighbors.KNeighborsClassifier(n_neighbors=2, p=1, weights='distance')
    cls.fit(X, y)
    y_prob = cls.predict_proba(np.array([[0, 2, 0], [2, 2, 2]]))
    real_prob = np.array([[0, 1, 0], [0, 0.4, 0.6]])
    assert_allclose(real_prob, y_prob)