from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.tests import test_dynamic_factor_mq_monte_carlo
def test_coefficient_of_determination(reset_randomstate, close_figures):
    endog, _, _, _, _, _ = gen_dfm_data(k_endog=3, nobs=1000)
    endog.iloc[0, 10:20] = np.nan
    endog.iloc[2, 15:25] = np.nan
    factors = {0: ['global', 'block'], 1: ['global', 'block'], 2: ['global']}
    mod = dynamic_factor_mq.DynamicFactorMQ(endog, factors=factors, standardize=False, idiosyncratic_ar1=False)
    res = mod.smooth(mod.start_params)
    factors = res.factors.smoothed
    actual = res.get_coefficients_of_determination(method='individual')
    desired = pd.DataFrame(np.zeros((3, 2)), index=[0, 1, 2], columns=['global', 'block'])
    for i in range(3):
        for j in range(2):
            if i == 2 and j == 1:
                desired.iloc[i, j] = np.nan
            else:
                y = endog.iloc[:, i]
                X = add_constant(factors.iloc[:, j])
                mod_ols = OLS(y, X, missing='drop')
                res_ols = mod_ols.fit()
                desired.iloc[i, j] = res_ols.rsquared
    assert_(actual.index.equals(desired.index))
    assert_(actual.columns.equals(desired.columns))
    assert_allclose(actual, desired)
    actual = res.get_coefficients_of_determination(method='joint')
    desired = pd.Series(np.zeros(3), index=[0, 1, 2])
    for i in range(3):
        y = endog.iloc[:, i]
        if i == 2:
            X = add_constant(factors.iloc[:, 0])
        else:
            X = add_constant(factors)
        mod_ols = OLS(y, X, missing='drop')
        res_ols = mod_ols.fit()
        desired.iloc[i] = res_ols.rsquared
    assert_(actual.index.equals(desired.index))
    assert_allclose(actual, desired)
    actual = res.get_coefficients_of_determination(method='cumulative')
    desired = pd.DataFrame(np.zeros((3, 2)), index=[0, 1, 2], columns=['global', 'block'])
    for i in range(3):
        for j in range(2):
            if i == 2 and j == 1:
                desired.iloc[i, j] = np.nan
            else:
                y = endog.iloc[:, i]
                X = add_constant(factors.iloc[:, :j + 1])
                mod_ols = OLS(y, X, missing='drop')
                res_ols = mod_ols.fit()
                desired.iloc[i, j] = res_ols.rsquared
    assert_(actual.index.equals(desired.index))
    assert_(actual.columns.equals(desired.columns))
    assert_allclose(actual, desired)
    factors = res.factors.filtered
    actual = res.get_coefficients_of_determination(method='individual', which='filtered')
    desired = pd.DataFrame(np.zeros((3, 2)), index=[0, 1, 2], columns=['global', 'block'])
    for i in range(3):
        for j in range(2):
            if i == 2 and j == 1:
                desired.iloc[i, j] = np.nan
            else:
                y = endog.iloc[:, i]
                X = add_constant(factors.iloc[:, j])
                mod_ols = OLS(y, X, missing='drop')
                res_ols = mod_ols.fit()
                desired.iloc[i, j] = res_ols.rsquared
    assert_allclose(actual, desired)
    try:
        import matplotlib.pyplot as plt
        try:
            from pandas.plotting import register_matplotlib_converters
            register_matplotlib_converters()
        except ImportError:
            pass
        fig1 = plt.figure()
        res.plot_coefficients_of_determination(method='individual', fig=fig1)
        fig2 = plt.figure()
        res.plot_coefficients_of_determination(method='joint', fig=fig2)
        fig3 = plt.figure()
        res.plot_coefficients_of_determination(method='cumulative', fig=fig3)
        fig4 = plt.figure()
        res.plot_coefficients_of_determination(which='filtered', fig=fig4)
    except ImportError:
        pass