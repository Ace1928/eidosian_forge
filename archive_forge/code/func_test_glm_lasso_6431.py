import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.testing import assert_series_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.compat.scipy import SP_LT_17
from statsmodels.datasets import cpunish, longley
from statsmodels.discrete import discrete_model as discrete
from statsmodels.genmod.generalized_linear_model import GLM, SET_USE_BIC_LLF
from statsmodels.tools.numdiff import (
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import add_constant
def test_glm_lasso_6431():
    np.random.seed(123)
    from statsmodels.regression.linear_model import OLS
    n = 50
    x = np.ones((n, 2))
    x[:, 1] = np.arange(0, n)
    y = 1000 + x[:, 1] + np.random.normal(0, 1, n)
    params = np.r_[999.82244338, 1.0077889]
    for method in ('bfgs', None):
        for fun in [OLS, GLM]:
            for L1_wtValue in [0, 1e-09]:
                model = fun(y, x)
                if fun == OLS:
                    fit = model.fit_regularized(alpha=0, L1_wt=L1_wtValue)
                else:
                    fit = model._fit_ridge(alpha=0, start_params=None, method=method)
                assert_allclose(params, fit.params, atol=1e-06, rtol=1e-06)