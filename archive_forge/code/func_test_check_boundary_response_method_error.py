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
def test_check_boundary_response_method_error():
    """Check that we raise an error for the cases not supported by
    `_check_boundary_response_method`.
    """

    class MultiLabelClassifier:
        classes_ = [np.array([0, 1]), np.array([0, 1])]
    err_msg = 'Multi-label and multi-output multi-class classifiers are not supported'
    with pytest.raises(ValueError, match=err_msg):
        _check_boundary_response_method(MultiLabelClassifier(), 'predict', None)

    class MulticlassClassifier:
        classes_ = [0, 1, 2]
    err_msg = 'Multiclass classifiers are only supported when `response_method` is'
    for response_method in ('predict_proba', 'decision_function'):
        with pytest.raises(ValueError, match=err_msg):
            _check_boundary_response_method(MulticlassClassifier(), response_method, None)