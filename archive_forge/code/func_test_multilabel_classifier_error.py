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
@pytest.mark.parametrize('response_method', ['auto', 'predict', 'predict_proba'])
def test_multilabel_classifier_error(pyplot, response_method):
    """Check that multilabel classifier raises correct error."""
    X, y = make_multilabel_classification(random_state=0)
    X = X[:, :2]
    tree = DecisionTreeClassifier().fit(X, y)
    msg = 'Multi-label and multi-output multi-class classifiers are not supported'
    with pytest.raises(ValueError, match=msg):
        DecisionBoundaryDisplay.from_estimator(tree, X, response_method=response_method)