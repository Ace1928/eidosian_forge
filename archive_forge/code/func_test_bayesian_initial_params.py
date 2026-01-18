from math import log
import numpy as np
import pytest
from sklearn import datasets
from sklearn.linear_model import ARDRegression, BayesianRidge, Ridge
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.extmath import fast_logdet
def test_bayesian_initial_params():
    X = np.vander(np.linspace(0, 4, 5), 4)
    y = np.array([0.0, 1.0, 0.0, -1.0, 0.0])
    reg = BayesianRidge(alpha_init=1.0, lambda_init=0.001)
    r2 = reg.fit(X, y).score(X, y)
    assert_almost_equal(r2, 1.0)