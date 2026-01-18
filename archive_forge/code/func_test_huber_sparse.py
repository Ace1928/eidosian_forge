import numpy as np
import pytest
from scipy import optimize
from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge, SGDRegressor
from sklearn.linear_model._huber import _huber_loss_and_gradient
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_huber_sparse(csr_container):
    X, y = make_regression_with_outliers()
    huber = HuberRegressor(alpha=0.1)
    huber.fit(X, y)
    X_csr = csr_container(X)
    huber_sparse = HuberRegressor(alpha=0.1)
    huber_sparse.fit(X_csr, y)
    assert_array_almost_equal(huber_sparse.coef_, huber.coef_)
    assert_array_equal(huber.outliers_, huber_sparse.outliers_)