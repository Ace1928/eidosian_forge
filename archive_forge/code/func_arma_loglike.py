import numpy as np
from statsmodels.tsa import arima_process
from statsmodels.tsa.statespace.tools import prefix_dtype_map
from statsmodels.tools.numdiff import _get_epsilon, approx_fprime_cs
from scipy.linalg.blas import find_best_blas_type
from . import _arma_innovations
def arma_loglike(endog, ar_params=None, ma_params=None, sigma2=1, prefix=None):
    """
    Compute the log-likelihood of the given data assuming an ARMA process.

    Parameters
    ----------
    endog : ndarray
        The observed time-series process.
    ar_params : ndarray, optional
        Autoregressive parameters.
    ma_params : ndarray, optional
        Moving average parameters.
    sigma2 : ndarray, optional
        The ARMA innovation variance. Default is 1.
    prefix : str, optional
        The BLAS prefix associated with the datatype. Default is to find the
        best datatype based on given input. This argument is typically only
        used internally.

    Returns
    -------
    float
        The joint loglikelihood.
    """
    llf_obs = arma_loglikeobs(endog, ar_params=ar_params, ma_params=ma_params, sigma2=sigma2, prefix=prefix)
    return np.sum(llf_obs)