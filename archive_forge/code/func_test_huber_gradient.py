import numpy as np
import pytest
from scipy import optimize
from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge, SGDRegressor
from sklearn.linear_model._huber import _huber_loss_and_gradient
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_huber_gradient():
    rng = np.random.RandomState(1)
    X, y = make_regression_with_outliers()
    sample_weight = rng.randint(1, 3, y.shape[0])

    def loss_func(x, *args):
        return _huber_loss_and_gradient(x, *args)[0]

    def grad_func(x, *args):
        return _huber_loss_and_gradient(x, *args)[1]
    for _ in range(5):
        for n_features in [X.shape[1] + 1, X.shape[1] + 2]:
            w = rng.randn(n_features)
            w[-1] = np.abs(w[-1])
            grad_same = optimize.check_grad(loss_func, grad_func, w, X, y, 0.01, 0.1, sample_weight)
            assert_almost_equal(grad_same, 1e-06, 4)