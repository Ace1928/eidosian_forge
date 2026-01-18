import numpy as np
import pytest
from sklearn.base import ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.linear_model import PassiveAggressiveClassifier, PassiveAggressiveRegressor
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_equal_class_weight():
    X2 = [[1, 0], [1, 0], [0, 1], [0, 1]]
    y2 = [0, 0, 1, 1]
    clf = PassiveAggressiveClassifier(C=0.1, tol=None, class_weight=None)
    clf.fit(X2, y2)
    clf_balanced = PassiveAggressiveClassifier(C=0.1, tol=None, class_weight='balanced')
    clf_balanced.fit(X2, y2)
    clf_weighted = PassiveAggressiveClassifier(C=0.1, tol=None, class_weight={0: 0.5, 1: 0.5})
    clf_weighted.fit(X2, y2)
    assert_almost_equal(clf.coef_, clf_weighted.coef_, decimal=2)
    assert_almost_equal(clf.coef_, clf_balanced.coef_, decimal=2)