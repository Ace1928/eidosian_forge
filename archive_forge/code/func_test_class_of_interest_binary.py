import warnings
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import (
from sklearn.ensemble import IsolationForest
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.inspection._plot.decision_boundary import _check_boundary_response_method
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._testing import (
@pytest.mark.parametrize('response_method', ['predict_proba', 'decision_function'])
def test_class_of_interest_binary(pyplot, response_method):
    """Check the behaviour of passing `class_of_interest` for plotting the output of
    `predict_proba` and `decision_function` in the binary case.
    """
    iris = load_iris()
    X = iris.data[:100, :2]
    y = iris.target[:100]
    assert_array_equal(np.unique(y), [0, 1])
    estimator = LogisticRegression().fit(X, y)
    disp_default = DecisionBoundaryDisplay.from_estimator(estimator, X, response_method=response_method, class_of_interest=None)
    disp_class_1 = DecisionBoundaryDisplay.from_estimator(estimator, X, response_method=response_method, class_of_interest=estimator.classes_[1])
    assert_allclose(disp_default.response, disp_class_1.response)
    disp_class_0 = DecisionBoundaryDisplay.from_estimator(estimator, X, response_method=response_method, class_of_interest=estimator.classes_[0])
    if response_method == 'predict_proba':
        assert_allclose(disp_default.response, 1 - disp_class_0.response)
    else:
        assert response_method == 'decision_function'
        assert_allclose(disp_default.response, -disp_class_0.response)