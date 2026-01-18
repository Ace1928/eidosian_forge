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
@pytest.mark.parametrize('response_method', ['predict', 'decision_function', ['decision_function', 'predict']])
@pytest.mark.parametrize('return_response_method_used', [True, False])
def test_get_response_values_outlier_detection(response_method, return_response_method_used):
    """Check the behaviour of `_get_response_values` with outlier detector."""
    X, y = make_classification(n_samples=50, random_state=0)
    outlier_detector = IsolationForest(random_state=0).fit(X, y)
    results = _get_response_values(outlier_detector, X, response_method=response_method, return_response_method_used=return_response_method_used)
    chosen_response_method = response_method[0] if isinstance(response_method, list) else response_method
    prediction_method = getattr(outlier_detector, chosen_response_method)
    assert_array_equal(results[0], prediction_method(X))
    assert results[1] is None
    if return_response_method_used:
        assert results[2] == chosen_response_method