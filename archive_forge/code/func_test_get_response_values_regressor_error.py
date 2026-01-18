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
@pytest.mark.parametrize('response_method', ['decision_function', 'predict_proba', 'predict_log_proba'])
def test_get_response_values_regressor_error(response_method):
    """Check the error message with regressor an not supported response
    method."""
    my_estimator = _MockEstimatorOnOffPrediction(response_methods=[response_method])
    X = ('mocking_data', 'mocking_target')
    err_msg = f'{my_estimator.__class__.__name__} should either be a classifier'
    with pytest.raises(ValueError, match=err_msg):
        _get_response_values(my_estimator, X, response_method=response_method)