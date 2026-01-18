import numpy as np
import pandas as pd
from statsmodels.multivariate.multivariate_ols import _MultivariateOLS
from numpy.testing import assert_array_almost_equal, assert_raises
import patsy
def test_independent_variable_singular():
    data1 = data.copy()
    data1['dup'] = data1['Drug']
    mod = _MultivariateOLS.from_formula('Histamine0 + Histamine1 + Histamine3 + Histamine5 ~ Drug * dup', data1)
    assert_raises(ValueError, mod.fit)
    mod = _MultivariateOLS.from_formula('Histamine0 + Histamine1 + Histamine3 + Histamine5 ~ Drug * dup', data1)
    assert_raises(ValueError, mod.fit)