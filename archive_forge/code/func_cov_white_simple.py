import numpy as np
from statsmodels.tools.grouputils import combine_indices, group_sums
from statsmodels.stats.moment_helpers import se_cov
def cov_white_simple(results, use_correction=True):
    """
    heteroscedasticity robust covariance matrix (White)

    Parameters
    ----------
    results : result instance
       result of a regression, uses results.model.exog and results.resid
       TODO: this should use wexog instead

    Returns
    -------
    cov : ndarray, (k_vars, k_vars)
        heteroscedasticity robust covariance matrix for parameter estimates

    Notes
    -----
    This produces the same result as cov_hc0, and does not include any small
    sample correction.

    verified (against LinearRegressionResults and Peterson)

    See Also
    --------
    cov_hc1, cov_hc2, cov_hc3 : heteroscedasticity robust covariance matrices
        with small sample corrections

    """
    xu, hessian_inv = _get_sandwich_arrays(results)
    sigma = S_white_simple(xu)
    cov_w = _HCCM2(hessian_inv, sigma)
    if use_correction:
        nobs, k_params = xu.shape
        cov_w *= nobs / float(nobs - k_params)
    return cov_w