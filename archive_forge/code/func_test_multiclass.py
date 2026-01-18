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
@pytest.mark.parametrize('response_method', ['auto', 'predict'])
def test_multiclass(pyplot, response_method):
    """Check multiclass gives expected results."""
    grid_resolution = 10
    eps = 1.0
    X, y = make_classification(n_classes=3, n_informative=3, random_state=0)
    X = X[:, [0, 1]]
    lr = LogisticRegression(random_state=0).fit(X, y)
    disp = DecisionBoundaryDisplay.from_estimator(lr, X, response_method=response_method, grid_resolution=grid_resolution, eps=1.0)
    x0_min, x0_max = (X[:, 0].min() - eps, X[:, 0].max() + eps)
    x1_min, x1_max = (X[:, 1].min() - eps, X[:, 1].max() + eps)
    xx0, xx1 = np.meshgrid(np.linspace(x0_min, x0_max, grid_resolution), np.linspace(x1_min, x1_max, grid_resolution))
    response = lr.predict(np.c_[xx0.ravel(), xx1.ravel()])
    assert_allclose(disp.response, response.reshape(xx0.shape))
    assert_allclose(disp.xx0, xx0)
    assert_allclose(disp.xx1, xx1)