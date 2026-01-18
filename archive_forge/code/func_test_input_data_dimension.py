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
def test_input_data_dimension(pyplot):
    """Check that we raise an error when `X` does not have exactly 2 features."""
    X, y = make_classification(n_samples=10, n_features=4, random_state=0)
    clf = LogisticRegression().fit(X, y)
    msg = 'n_features must be equal to 2. Got 4 instead.'
    with pytest.raises(ValueError, match=msg):
        DecisionBoundaryDisplay.from_estimator(estimator=clf, X=X)