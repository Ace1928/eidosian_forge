import numpy as np
from statsmodels.regression.linear_model import yule_walker
from statsmodels.stats.moment_helpers import cov2corr
def corr2cov(corr, std):
    """convert correlation matrix to covariance matrix

    Parameters
    ----------
    corr : ndarray, (k_vars, k_vars)
        correlation matrix
    std : ndarray, (k_vars,) or scalar
        standard deviation for the vector of random variables. If scalar, then
        it is assumed that all variables have the same scale given by std.

    """
    if np.size(std) == 1:
        std = std * np.ones(corr.shape[0])
    cov = corr * std[:, None] * std[None, :]
    return cov