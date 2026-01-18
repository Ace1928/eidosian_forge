import numpy as np
import pandas as pd
from statsmodels.multivariate.multivariate_ols import _MultivariateOLS
from numpy.testing import assert_array_almost_equal, assert_raises
import patsy
def test_exog_1D_array():
    mod = _MultivariateOLS.from_formula('Histamine0 + Histamine1 + Histamine3 + Histamine5 ~ 0 + Depleted', data)
    r = mod.fit(method='svd')
    r0 = r.mv_test()
    a = [[0.0019, 8.0, 20.0, 55.0013, 0.0], [1.8112, 8.0, 22.0, 26.3796, 0.0], [97.8858, 8.0, 12.1818, 117.1133, 0.0], [93.2742, 4.0, 11.0, 256.5041, 0.0]]
    assert_array_almost_equal(r0['Depleted']['stat'].values, a, decimal=4)