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
def test_get_response_values_with_response_list():
    """Check the behaviour of passing a list of responses to `_get_response_values`."""
    classifier = LogisticRegression().fit(X_binary, y_binary)
    y_pred, pos_label, response_method = _get_response_values(classifier, X_binary, response_method=['predict_proba', 'decision_function'], return_response_method_used=True)
    assert_allclose(y_pred, classifier.predict_proba(X_binary)[:, 1])
    assert pos_label == 1
    assert response_method == 'predict_proba'
    y_pred, pos_label, response_method = _get_response_values(classifier, X_binary, response_method=['decision_function', 'predict_proba'], return_response_method_used=True)
    assert_allclose(y_pred, classifier.decision_function(X_binary))
    assert pos_label == 1
    assert response_method == 'decision_function'