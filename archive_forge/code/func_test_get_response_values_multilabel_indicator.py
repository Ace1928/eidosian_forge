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
@pytest.mark.parametrize('response_method', ['predict_proba', 'decision_function', 'predict'])
def test_get_response_values_multilabel_indicator(response_method):
    X, Y = make_multilabel_classification(random_state=0)
    estimator = ClassifierChain(LogisticRegression()).fit(X, Y)
    y_pred, pos_label = _get_response_values(estimator, X, response_method=response_method)
    assert pos_label is None
    assert y_pred.shape == Y.shape
    if response_method == 'predict_proba':
        assert np.logical_and(y_pred >= 0, y_pred <= 1).all()
    elif response_method == 'decision_function':
        assert (y_pred < 0).sum() > 0
        assert (y_pred > 1).sum() > 0
    else:
        assert np.logical_or(y_pred == 0, y_pred == 1).all()