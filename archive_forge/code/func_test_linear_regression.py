import warnings
import numpy as np
import pytest
from scipy import linalg, sparse
from sklearn.datasets import load_iris, make_regression, make_sparse_uncorrelated
from sklearn.linear_model import LinearRegression
from sklearn.linear_model._base import (
from sklearn.preprocessing import add_dummy_feature
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
def test_linear_regression():
    X = [[1], [2]]
    Y = [1, 2]
    reg = LinearRegression()
    reg.fit(X, Y)
    assert_array_almost_equal(reg.coef_, [1])
    assert_array_almost_equal(reg.intercept_, [0])
    assert_array_almost_equal(reg.predict(X), [1, 2])
    X = [[1]]
    Y = [0]
    reg = LinearRegression()
    reg.fit(X, Y)
    assert_array_almost_equal(reg.coef_, [0])
    assert_array_almost_equal(reg.intercept_, [0])
    assert_array_almost_equal(reg.predict(X), [0])