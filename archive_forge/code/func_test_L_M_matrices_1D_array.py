import numpy as np
import pandas as pd
from statsmodels.multivariate.multivariate_ols import _MultivariateOLS
from numpy.testing import assert_array_almost_equal, assert_raises
import patsy
def test_L_M_matrices_1D_array():
    mod = _MultivariateOLS.from_formula('Histamine0 + Histamine1 + Histamine3 + Histamine5 ~ Drug * Depleted', data)
    r = mod.fit(method='svd')
    L = np.array([1, 0, 0, 0, 0, 0])
    assert_raises(ValueError, r.mv_test, hypotheses=[['Drug', L, None]])
    L = np.array([[1, 0, 0, 0, 0, 0]])
    M = np.array([1, 0, 0, 0, 0, 0])
    assert_raises(ValueError, r.mv_test, hypotheses=[['Drug', L, M]])