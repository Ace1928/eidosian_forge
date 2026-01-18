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
def test_fit_intercept():
    X2 = np.array([[0.38349978, 0.61650022], [0.58853682, 0.41146318]])
    X3 = np.array([[0.27677969, 0.70693172, 0.01628859], [0.08385139, 0.20692515, 0.70922346]])
    y = np.array([1, 1])
    lr2_without_intercept = LinearRegression(fit_intercept=False).fit(X2, y)
    lr2_with_intercept = LinearRegression().fit(X2, y)
    lr3_without_intercept = LinearRegression(fit_intercept=False).fit(X3, y)
    lr3_with_intercept = LinearRegression().fit(X3, y)
    assert lr2_with_intercept.coef_.shape == lr2_without_intercept.coef_.shape
    assert lr3_with_intercept.coef_.shape == lr3_without_intercept.coef_.shape
    assert lr2_without_intercept.coef_.ndim == lr3_without_intercept.coef_.ndim