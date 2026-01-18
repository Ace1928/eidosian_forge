import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import svds
from scipy.optimize import fminbound
import warnings
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import (
def cov_nearest(cov, method='clipped', threshold=1e-15, n_fact=100, return_all=False):
    """
    Find the nearest covariance matrix that is positive (semi-) definite

    This leaves the diagonal, i.e. the variance, unchanged

    Parameters
    ----------
    cov : ndarray, (k,k)
        initial covariance matrix
    method : str
        if "clipped", then the faster but less accurate ``corr_clipped`` is
        used.if "nearest", then ``corr_nearest`` is used
    threshold : float
        clipping threshold for smallest eigen value, see Notes
    n_fact : int or float
        factor to determine the maximum number of iterations in
        ``corr_nearest``. See its doc string
    return_all : bool
        if False (default), then only the covariance matrix is returned.
        If True, then correlation matrix and standard deviation are
        additionally returned.

    Returns
    -------
    cov_ : ndarray
        corrected covariance matrix
    corr_ : ndarray, (optional)
        corrected correlation matrix
    std_ : ndarray, (optional)
        standard deviation


    Notes
    -----
    This converts the covariance matrix to a correlation matrix. Then, finds
    the nearest correlation matrix that is positive semidefinite and converts
    it back to a covariance matrix using the initial standard deviation.

    The smallest eigenvalue of the intermediate correlation matrix is
    approximately equal to the ``threshold``.
    If the threshold=0, then the smallest eigenvalue of the correlation matrix
    might be negative, but zero within a numerical error, for example in the
    range of -1e-16.

    Assumes input covariance matrix is symmetric.

    See Also
    --------
    corr_nearest
    corr_clipped
    """
    from statsmodels.stats.moment_helpers import cov2corr, corr2cov
    cov_, std_ = cov2corr(cov, return_std=True)
    if method == 'clipped':
        corr_ = corr_clipped(cov_, threshold=threshold)
    else:
        corr_ = corr_nearest(cov_, threshold=threshold, n_fact=n_fact)
    cov_ = corr2cov(corr_, std_)
    if return_all:
        return (cov_, corr_, std_)
    else:
        return cov_