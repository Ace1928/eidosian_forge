import numpy as np
import pytest
from sklearn.base import ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.linear_model import PassiveAggressiveClassifier, PassiveAggressiveRegressor
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', [None, *CSR_CONTAINERS])
@pytest.mark.parametrize('loss', ('hinge', 'squared_hinge'))
def test_classifier_correctness(loss, csr_container):
    y_bin = y.copy()
    y_bin[y != 1] = -1
    clf1 = MyPassiveAggressive(loss=loss, n_iter=2)
    clf1.fit(X, y_bin)
    data = csr_container(X) if csr_container is not None else X
    clf2 = PassiveAggressiveClassifier(loss=loss, max_iter=2, shuffle=False, tol=None)
    clf2.fit(data, y_bin)
    assert_array_almost_equal(clf1.w, clf2.coef_.ravel(), decimal=2)