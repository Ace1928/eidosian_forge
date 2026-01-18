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
def test_get_response_decision_function():
    """Check the behaviour of `_get_response_values_binary` using decision_function."""
    classifier = LogisticRegression().fit(X_binary, y_binary)
    y_score, pos_label = _get_response_values_binary(classifier, X_binary, response_method='decision_function')
    assert_allclose(y_score, classifier.decision_function(X_binary))
    assert pos_label == 1
    y_score, pos_label = _get_response_values_binary(classifier, X_binary, response_method='decision_function', pos_label=0)
    assert_allclose(y_score, classifier.decision_function(X_binary) * -1)
    assert pos_label == 0