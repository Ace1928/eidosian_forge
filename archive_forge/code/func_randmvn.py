from pystatsmodels mailinglist 20100524
from collections import namedtuple
from statsmodels.compat.python import lzip, lrange
import copy
import math
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from scipy import stats, interpolate
from statsmodels.iolib.table import SimpleTable
from statsmodels.stats.multitest import multipletests, _ecdf as ecdf, fdrcorrection as fdrcorrection0, fdrcorrection_twostage
from statsmodels.graphics import utils
from statsmodels.tools.sm_exceptions import ValueWarning
def randmvn(rho, size=(1, 2), standardize=False):
    """create random draws from equi-correlated multivariate normal distribution

    Parameters
    ----------
    rho : float
        correlation coefficient
    size : tuple of int
        size is interpreted (nobs, nvars) where each row

    Returns
    -------
    rvs : ndarray
        nobs by nvars where each row is a independent random draw of nvars-
        dimensional correlated rvs

    """
    nobs, nvars = size
    if 0 < rho and rho < 1:
        rvs = np.random.randn(nobs, nvars + 1)
        rvs2 = rvs[:, :-1] * np.sqrt(1 - rho) + rvs[:, -1:] * np.sqrt(rho)
    elif rho == 0:
        rvs2 = np.random.randn(nobs, nvars)
    elif rho < 0:
        if rho < -1.0 / (nvars - 1):
            raise ValueError('rho has to be larger than -1./(nvars-1)')
        elif rho == -1.0 / (nvars - 1):
            rho = -1.0 / (nvars - 1 + 1e-10)
        A = rho * np.ones((nvars, nvars)) + (1 - rho) * np.eye(nvars)
        rvs2 = np.dot(np.random.randn(nobs, nvars), np.linalg.cholesky(A).T)
    if standardize:
        rvs2 = stats.zscore(rvs2)
    return rvs2