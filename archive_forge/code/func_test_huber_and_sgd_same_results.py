import numpy as np
import pytest
from scipy import optimize
from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge, SGDRegressor
from sklearn.linear_model._huber import _huber_loss_and_gradient
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_huber_and_sgd_same_results():
    X, y = make_regression_with_outliers(n_samples=10, n_features=2)
    huber = HuberRegressor(fit_intercept=False, alpha=0.0, epsilon=1.35)
    huber.fit(X, y)
    X_scale = X / huber.scale_
    y_scale = y / huber.scale_
    huber.fit(X_scale, y_scale)
    assert_almost_equal(huber.scale_, 1.0, 3)
    sgdreg = SGDRegressor(alpha=0.0, loss='huber', shuffle=True, random_state=0, max_iter=10000, fit_intercept=False, epsilon=1.35, tol=None)
    sgdreg.fit(X_scale, y_scale)
    assert_array_almost_equal(huber.coef_, sgdreg.coef_, 1)