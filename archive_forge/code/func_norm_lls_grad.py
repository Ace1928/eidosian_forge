import numpy as np
from scipy import special
from scipy.special import gammaln
def norm_lls_grad(y, params):
    """Jacobian of normal loglikelihood wrt mean mu and variance sigma2

    Parameters
    ----------
    y : ndarray, 1d
        normally distributed random variable
    params : ndarray, (nobs, 2)
        array of mean, variance (mu, sigma2) with observations in rows

    Returns
    -------
    grad : array (nobs, 2)
        derivative of loglikelihood for each observation wrt mean in first
        column, and wrt variance in second column

    Notes
    -----
    this is actually the derivative wrt sigma not sigma**2, but evaluated
    with parameter sigma2 = sigma**2

    """
    mu, sigma2 = params.T
    dllsdmu = (y - mu) / sigma2
    dllsdsigma2 = ((y - mu) ** 2 / sigma2 - 1) / np.sqrt(sigma2)
    return np.column_stack((dllsdmu, dllsdsigma2))