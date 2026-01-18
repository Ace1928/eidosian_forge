from math import log
import numpy as np
import pytest
from sklearn import datasets
from sklearn.linear_model import ARDRegression, BayesianRidge, Ridge
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.extmath import fast_logdet
@pytest.mark.parametrize('Estimator', [BayesianRidge, ARDRegression])
def test_dtype_correctness(Estimator):
    X = np.array([[1, 1], [3, 4], [5, 7], [4, 1], [2, 6], [3, 10], [3, 2]])
    y = np.array([1, 2, 3, 2, 0, 4, 5]).T
    model = Estimator()
    coef_32 = model.fit(X.astype(np.float32), y).coef_
    coef_64 = model.fit(X.astype(np.float64), y).coef_
    np.testing.assert_allclose(coef_32, coef_64, rtol=0.0001)