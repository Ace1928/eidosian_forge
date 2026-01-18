from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.tests import test_dynamic_factor_mq_monte_carlo
def gen_dfm_data(k_endog=2, nobs=1000):
    if k_endog > 10:
        raise ValueError('Only allows for k_endog <= 10')
    ix = pd.period_range(start='1950-01', periods=1, freq='M')
    faux = pd.DataFrame([[0] * k_endog], index=ix)
    mod = dynamic_factor.DynamicFactor(faux, k_factors=1, factor_order=1)
    loadings = [0.5, -0.9, 0.2, 0.7, -0.1, -0.1, 0.4, 0.4, 0.8, 0.8][:k_endog]
    phi = 0.5
    sigma2 = 1.0
    idio_ar1 = [0] * k_endog
    idio_var = [1.0, 0.2, 1.5, 0.8, 0.8, 1.4, 0.1, 0.2, 0.4, 0.5][:k_endog]
    params = np.r_[loadings, idio_var, phi]
    endog = mod.simulate(params, nobs)
    return (endog, loadings, phi, sigma2, idio_ar1, idio_var)