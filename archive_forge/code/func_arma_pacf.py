from statsmodels.compat.numpy import NP_LT_2
from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from scipy import linalg, optimize, signal
from statsmodels.tools.docstring import Docstring, remove_parameters
from statsmodels.tools.validation import array_like
def arma_pacf(ar, ma, lags=10):
    """
    Theoretical partial autocorrelation function of an ARMA process.

    Parameters
    ----------
    ar : array_like, 1d
        The coefficients for autoregressive lag polynomial, including zero lag.
    ma : array_like, 1d
        The coefficients for moving-average lag polynomial, including zero lag.
    lags : int
        The number of terms (lags plus zero lag) to include in returned pacf.

    Returns
    -------
    ndarrray
        The partial autocorrelation of ARMA process given by ar and ma.

    Notes
    -----
    Solves yule-walker equation for each lag order up to nobs lags.

    not tested/checked yet
    """
    apacf = np.zeros(lags)
    acov = arma_acf(ar, ma, lags=lags + 1)
    apacf[0] = 1.0
    for k in range(2, lags + 1):
        r = acov[:k]
        apacf[k - 1] = linalg.solve(linalg.toeplitz(r[:-1]), r[1:])[-1]
    return apacf