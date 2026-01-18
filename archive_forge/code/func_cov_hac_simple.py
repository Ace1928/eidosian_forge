import numpy as np
from statsmodels.tools.grouputils import combine_indices, group_sums
from statsmodels.stats.moment_helpers import se_cov
def cov_hac_simple(results, nlags=None, weights_func=weights_bartlett, use_correction=True):
    """
    heteroscedasticity and autocorrelation robust covariance matrix (Newey-West)

    Assumes we have a single time series with zero axis consecutive, equal
    spaced time periods


    Parameters
    ----------
    results : result instance
       result of a regression, uses results.model.exog and results.resid
       TODO: this should use wexog instead
    nlags : int or None
        highest lag to include in kernel window. If None, then
        nlags = floor[4(T/100)^(2/9)] is used.
    weights_func : callable
        weights_func is called with nlags as argument to get the kernel
        weights. default are Bartlett weights

    Returns
    -------
    cov : ndarray, (k_vars, k_vars)
        HAC robust covariance matrix for parameter estimates

    Notes
    -----
    verified only for nlags=0, which is just White
    just guessing on correction factor, need reference

    options might change when other kernels besides Bartlett are available.

    """
    xu, hessian_inv = _get_sandwich_arrays(results)
    sigma = S_hac_simple(xu, nlags=nlags, weights_func=weights_func)
    cov_hac = _HCCM2(hessian_inv, sigma)
    if use_correction:
        nobs, k_params = xu.shape
        cov_hac *= nobs / float(nobs - k_params)
    return cov_hac