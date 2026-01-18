from math import log
import numpy as np
import pytest
from sklearn import datasets
from sklearn.linear_model import ARDRegression, BayesianRidge, Ridge
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.extmath import fast_logdet
def test_bayesian_ridge_parameter():
    X = np.array([[1, 1], [3, 4], [5, 7], [4, 1], [2, 6], [3, 10], [3, 2]])
    y = np.array([1, 2, 3, 2, 0, 4, 5]).T
    br_model = BayesianRidge(compute_score=True).fit(X, y)
    rr_model = Ridge(alpha=br_model.lambda_ / br_model.alpha_).fit(X, y)
    assert_array_almost_equal(rr_model.coef_, br_model.coef_)
    assert_almost_equal(rr_model.intercept_, br_model.intercept_)