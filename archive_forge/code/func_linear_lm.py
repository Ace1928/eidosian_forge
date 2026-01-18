from statsmodels.compat.pandas import deprecate_kwarg
from collections.abc import Iterable
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS, RegressionResultsWrapper
from statsmodels.stats._adnorm import anderson_statistic, normal_ad
from statsmodels.stats._lilliefors import (
from statsmodels.tools.validation import (
from statsmodels.tsa.tsatools import lagmat
def linear_lm(resid, exog, func=None):
    """
    Lagrange multiplier test for linearity against functional alternative

    # TODO: Remove the restriction
    limitations: Assumes currently that the first column is integer.
    Currently it does not check whether the transformed variables contain NaNs,
    for example log of negative number.

    Parameters
    ----------
    resid : ndarray
        residuals of a regression
    exog : ndarray
        exogenous variables for which linearity is tested
    func : callable, default None
        If func is None, then squares are used. func needs to take an array
        of exog and return an array of transformed variables.

    Returns
    -------
    lm : float
       Lagrange multiplier test statistic
    lm_pval : float
       p-value of Lagrange multiplier tes
    ftest : ContrastResult instance
       the results from the F test variant of this test

    Notes
    -----
    Written to match Gretl's linearity test. The test runs an auxiliary
    regression of the residuals on the combined original and transformed
    regressors. The Null hypothesis is that the linear specification is
    correct.
    """
    if func is None:

        def func(x):
            return np.power(x, 2)
    exog = np.asarray(exog)
    exog_aux = np.column_stack((exog, func(exog[:, 1:])))
    nobs, k_vars = exog.shape
    ls = OLS(resid, exog_aux).fit()
    ftest = ls.f_test(np.eye(k_vars - 1, k_vars * 2 - 1, k_vars))
    lm = nobs * ls.rsquared
    lm_pval = stats.chi2.sf(lm, k_vars - 1)
    return (lm, lm_pval, ftest)