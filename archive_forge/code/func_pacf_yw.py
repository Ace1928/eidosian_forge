from __future__ import annotations
from statsmodels.compat.numpy import lstsq
from statsmodels.compat.pandas import deprecate_kwarg
from statsmodels.compat.python import lzip
from statsmodels.compat.scipy import _next_regular
from typing import Literal, Union
import warnings
import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import correlate
from statsmodels.regression.linear_model import OLS, yule_walker
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import Bunch, add_constant
from statsmodels.tools.validation import (
from statsmodels.tsa._bds import bds
from statsmodels.tsa._innovations import innovations_algo, innovations_filter
from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp
from statsmodels.tsa.tsatools import add_trend, lagmat, lagmat2ds
def pacf_yw(x: ArrayLike1D, nlags: int | None=None, method: Literal['adjusted', 'mle']='adjusted') -> np.ndarray:
    """
    Partial autocorrelation estimated with non-recursive yule_walker.

    Parameters
    ----------
    x : array_like
        The observations of time series for which pacf is calculated.
    nlags : int, optional
        Number of lags to return autocorrelation for. If not provided,
        uses min(10 * np.log10(nobs), nobs - 1).
    method : {"adjusted", "mle"}, default "adjusted"
        The method for the autocovariance calculations in yule walker.

    Returns
    -------
    ndarray
        The partial autocorrelations, maxlag+1 elements.

    See Also
    --------
    statsmodels.tsa.stattools.pacf
        Partial autocorrelation estimation.
    statsmodels.tsa.stattools.pacf_ols
        Partial autocorrelation estimation using OLS.
    statsmodels.tsa.stattools.pacf_burg
        Partial autocorrelation estimation using Burg"s method.

    Notes
    -----
    This solves yule_walker for each desired lag and contains
    currently duplicate calculations.
    """
    x = array_like(x, 'x')
    nlags = int_like(nlags, 'nlags', optional=True)
    nobs = x.shape[0]
    if nlags is None:
        nlags = max(min(int(10 * np.log10(nobs)), nobs - 1), 1)
    method = string_like(method, 'method', options=('adjusted', 'mle'))
    pacf = [1.0]
    with warnings.catch_warnings():
        warnings.simplefilter('once', ValueWarning)
        for k in range(1, nlags + 1):
            pacf.append(yule_walker(x, k, method=method)[0][-1])
    return np.array(pacf)