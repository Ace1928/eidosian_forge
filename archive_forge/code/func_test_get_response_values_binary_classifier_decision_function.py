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
def test_get_response_values_binary_classifier_decision_function(return_response_method_used):
    """Check the behaviour of `_get_response_values` with `decision_function`
    and binary classifier."""
    X, y = make_classification(n_samples=10, n_classes=2, weights=[0.3, 0.7], random_state=0)
    classifier = LogisticRegression().fit(X, y)
    response_method = 'decision_function'
    results = _get_response_values(classifier, X, response_method=response_method, pos_label=None, return_response_method_used=return_response_method_used)
    assert_allclose(results[0], classifier.decision_function(X))
    assert results[1] == 1
    if return_response_method_used:
        assert results[2] == 'decision_function'
    results = _get_response_values(classifier, X, response_method=response_method, pos_label=classifier.classes_[0], return_response_method_used=return_response_method_used)
    assert_allclose(results[0], classifier.decision_function(X) * -1)
    assert results[1] == 0
    if return_response_method_used:
        assert results[2] == 'decision_function'