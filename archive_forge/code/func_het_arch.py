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
@deprecate_kwarg('maxlag', 'nlags')
def het_arch(resid, nlags=None, store=False, ddof=0):
    """
    Engle's Test for Autoregressive Conditional Heteroscedasticity (ARCH).

    Parameters
    ----------
    resid : ndarray
        residuals from an estimation, or time series
    nlags : int, default None
        Highest lag to use.
    store : bool, default False
        If true then the intermediate results are also returned
    ddof : int, default 0
        If the residuals are from a regression, or ARMA estimation, then there
        are recommendations to correct the degrees of freedom by the number
        of parameters that have been estimated, for example ddof=p+q for an
        ARMA(p,q).

    Returns
    -------
    lm : float
        Lagrange multiplier test statistic
    lmpval : float
        p-value for Lagrange multiplier test
    fval : float
        fstatistic for F test, alternative version of the same test based on
        F test for the parameter restriction
    fpval : float
        pvalue for F test
    res_store : ResultsStore, optional
        Intermediate results. Returned if store is True.

    Notes
    -----
    verified against R:FinTS::ArchTest
    """
    return acorr_lm(resid ** 2, nlags=nlags, store=store, ddof=ddof)