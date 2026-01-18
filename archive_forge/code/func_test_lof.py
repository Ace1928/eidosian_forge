import re
from math import sqrt
import numpy as np
import pytest
from sklearn import metrics, neighbors
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_lof(global_dtype):
    X = np.asarray([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [5, 3], [-4, 2]], dtype=global_dtype)
    clf = neighbors.LocalOutlierFactor(n_neighbors=5)
    score = clf.fit(X).negative_outlier_factor_
    assert_array_equal(clf._fit_X, X)
    assert np.min(score[:-2]) > np.max(score[-2:])
    clf = neighbors.LocalOutlierFactor(contamination=0.25, n_neighbors=5).fit(X)
    expected_predictions = 6 * [1] + 2 * [-1]
    assert_array_equal(clf._predict(), expected_predictions)
    assert_array_equal(clf.fit_predict(X), expected_predictions)