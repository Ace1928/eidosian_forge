import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from sklearn.datasets import load_iris
from sklearn.utils import _safe_indexing, check_array
from sklearn.utils._mocking import (
from sklearn.utils._testing import _convert_container
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('input_type', ['list', 'array', 'sparse', 'dataframe'])
def test_checking_classifier(iris, input_type):
    X, y = iris
    X = _convert_container(X, input_type)
    clf = CheckingClassifier()
    clf.fit(X, y)
    assert_array_equal(clf.classes_, np.unique(y))
    assert len(clf.classes_) == 3
    assert clf.n_features_in_ == 4
    y_pred = clf.predict(X)
    assert_array_equal(y_pred, np.zeros(y_pred.size, dtype=int))
    assert clf.score(X) == pytest.approx(0)
    clf.set_params(foo_param=10)
    assert clf.fit(X, y).score(X) == pytest.approx(1)
    y_proba = clf.predict_proba(X)
    assert y_proba.shape == (150, 3)
    assert_allclose(y_proba[:, 0], 1)
    assert_allclose(y_proba[:, 1:], 0)
    y_decision = clf.decision_function(X)
    assert y_decision.shape == (150, 3)
    assert_allclose(y_decision[:, 0], 1)
    assert_allclose(y_decision[:, 1:], 0)
    first_2_classes = np.logical_or(y == 0, y == 1)
    X = _safe_indexing(X, first_2_classes)
    y = _safe_indexing(y, first_2_classes)
    clf.fit(X, y)
    y_proba = clf.predict_proba(X)
    assert y_proba.shape == (100, 2)
    assert_allclose(y_proba[:, 0], 1)
    assert_allclose(y_proba[:, 1], 0)
    y_decision = clf.decision_function(X)
    assert y_decision.shape == (100,)
    assert_allclose(y_decision, 0)