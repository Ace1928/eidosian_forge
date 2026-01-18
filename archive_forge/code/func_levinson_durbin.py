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
def levinson_durbin(s, nlags=10, isacov=False):
    """
    Levinson-Durbin recursion for autoregressive processes.

    Parameters
    ----------
    s : array_like
        If isacov is False, then this is the time series. If iasacov is true
        then this is interpreted as autocovariance starting with lag 0.
    nlags : int, optional
        The largest lag to include in recursion or order of the autoregressive
        process.
    isacov : bool, optional
        Flag indicating whether the first argument, s, contains the
        autocovariances or the data series.

    Returns
    -------
    sigma_v : float
        The estimate of the error variance.
    arcoefs : ndarray
        The estimate of the autoregressive coefficients for a model including
        nlags.
    pacf : ndarray
        The partial autocorrelation function.
    sigma : ndarray
        The entire sigma array from intermediate result, last value is sigma_v.
    phi : ndarray
        The entire phi array from intermediate result, last column contains
        autoregressive coefficients for AR(nlags).

    Notes
    -----
    This function returns currently all results, but maybe we drop sigma and
    phi from the returns.

    If this function is called with the time series (isacov=False), then the
    sample autocovariance function is calculated with the default options
    (biased, no fft).
    """
    s = array_like(s, 's')
    nlags = int_like(nlags, 'nlags')
    isacov = bool_like(isacov, 'isacov')
    order = nlags
    if isacov:
        sxx_m = s
    else:
        sxx_m = acovf(s, fft=False)[:order + 1]
    phi = np.zeros((order + 1, order + 1), 'd')
    sig = np.zeros(order + 1)
    phi[1, 1] = sxx_m[1] / sxx_m[0]
    sig[1] = sxx_m[0] - phi[1, 1] * sxx_m[1]
    for k in range(2, order + 1):
        phi[k, k] = (sxx_m[k] - np.dot(phi[1:k, k - 1], sxx_m[1:k][::-1])) / sig[k - 1]
        for j in range(1, k):
            phi[j, k] = phi[j, k - 1] - phi[k, k] * phi[k - j, k - 1]
        sig[k] = sig[k - 1] * (1 - phi[k, k] ** 2)
    sigma_v = sig[-1]
    arcoefs = phi[1:, -1]
    pacf_ = np.diag(phi).copy()
    pacf_[0] = 1.0
    return (sigma_v, arcoefs, pacf_, sig, phi)