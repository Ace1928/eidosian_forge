import numpy as np
import pytest
from sklearn.datasets import (
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import (
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._mocking import _MockEstimatorOnOffPrediction
from sklearn.utils._response import _get_response_values, _get_response_values_binary
from sklearn.utils._testing import assert_allclose, assert_array_equal
@pytest.mark.parametrize('return_response_method_used', [True, False])
@pytest.mark.parametrize('response_method', ['predict_proba', 'predict_log_proba'])
def test_get_response_values_binary_classifier_predict_proba(return_response_method_used, response_method):
    """Check that `_get_response_values` with `predict_proba` and binary
    classifier."""
    X, y = make_classification(n_samples=10, n_classes=2, weights=[0.3, 0.7], random_state=0)
    classifier = LogisticRegression().fit(X, y)
    results = _get_response_values(classifier, X, response_method=response_method, pos_label=None, return_response_method_used=return_response_method_used)
    assert_allclose(results[0], getattr(classifier, response_method)(X)[:, 1])
    assert results[1] == 1
    if return_response_method_used:
        assert len(results) == 3
        assert results[2] == response_method
    else:
        assert len(results) == 2
    y_pred, pos_label, *_ = _get_response_values(classifier, X, response_method=response_method, pos_label=classifier.classes_[0], return_response_method_used=return_response_method_used)
    assert_allclose(y_pred, getattr(classifier, response_method)(X)[:, 0])
    assert pos_label == 0