import numpy as np
import pytest
from sklearn.base import ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.linear_model import PassiveAggressiveClassifier, PassiveAggressiveRegressor
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_classifier_refit():
    clf = PassiveAggressiveClassifier(max_iter=5).fit(X, y)
    assert_array_equal(clf.classes_, np.unique(y))
    clf.fit(X[:, :-1], iris.target_names[y])
    assert_array_equal(clf.classes_, iris.target_names)