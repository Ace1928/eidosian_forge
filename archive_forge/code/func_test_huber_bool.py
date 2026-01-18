import numpy as np
import pytest
from scipy import optimize
from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge, SGDRegressor
from sklearn.linear_model._huber import _huber_loss_and_gradient
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_huber_bool():
    X, y = make_regression(n_samples=200, n_features=2, noise=4.0, random_state=0)
    X_bool = X > 0
    HuberRegressor().fit(X_bool, y)