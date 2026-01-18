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
@pytest.mark.parametrize('estimator, X, y, err_msg, params', [(DecisionTreeRegressor(), X_binary, y_binary, "Expected 'estimator' to be a binary classifier", {'response_method': 'auto'}), (DecisionTreeClassifier(), X_binary, y_binary, 'pos_label=unknown is not a valid label: It should be one of \\[0 1\\]', {'response_method': 'auto', 'pos_label': 'unknown'}), (DecisionTreeClassifier(), X, y, 'be a binary classifier. Got 3 classes instead.', {'response_method': 'predict_proba'})])
def test_get_response_error(estimator, X, y, err_msg, params):
    """Check that we raise the proper error messages in _get_response_values_binary."""
    estimator.fit(X, y)
    with pytest.raises(ValueError, match=err_msg):
        _get_response_values_binary(estimator, X, **params)