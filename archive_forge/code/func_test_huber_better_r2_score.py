import numpy as np
import pytest
from scipy import optimize
from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge, SGDRegressor
from sklearn.linear_model._huber import _huber_loss_and_gradient
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_huber_better_r2_score():
    X, y = make_regression_with_outliers()
    huber = HuberRegressor(alpha=0.01)
    huber.fit(X, y)
    linear_loss = np.dot(X, huber.coef_) + huber.intercept_ - y
    mask = np.abs(linear_loss) < huber.epsilon * huber.scale_
    huber_score = huber.score(X[mask], y[mask])
    huber_outlier_score = huber.score(X[~mask], y[~mask])
    ridge = Ridge(alpha=0.01)
    ridge.fit(X, y)
    ridge_score = ridge.score(X[mask], y[mask])
    ridge_outlier_score = ridge.score(X[~mask], y[~mask])
    assert huber_score > ridge_score
    assert ridge_outlier_score > huber_outlier_score