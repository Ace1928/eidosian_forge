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
def test_lof_values(global_dtype):
    X_train = np.asarray([[1, 1], [1, 2], [2, 1]], dtype=global_dtype)
    clf1 = neighbors.LocalOutlierFactor(n_neighbors=2, contamination=0.1, novelty=True).fit(X_train)
    clf2 = neighbors.LocalOutlierFactor(n_neighbors=2, novelty=True).fit(X_train)
    s_0 = 2.0 * sqrt(2.0) / (1.0 + sqrt(2.0))
    s_1 = (1.0 + sqrt(2)) * (1.0 / (4.0 * sqrt(2.0)) + 1.0 / (2.0 + 2.0 * sqrt(2)))
    assert_allclose(-clf1.negative_outlier_factor_, [s_0, s_1, s_1])
    assert_allclose(-clf2.negative_outlier_factor_, [s_0, s_1, s_1])
    assert_allclose(-clf1.score_samples([[2.0, 2.0]]), [s_0])
    assert_allclose(-clf2.score_samples([[2.0, 2.0]]), [s_0])
    assert_allclose(-clf1.score_samples([[1.0, 1.0]]), [s_1])
    assert_allclose(-clf2.score_samples([[1.0, 1.0]]), [s_1])