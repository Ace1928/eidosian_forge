from functools import reduce
import numpy as np
from statsmodels.regression.linear_model import GLS
from pandas import Panel
def repanel_cov(groups, sigmas):
    """calculate error covariance matrix for random effects model

    Parameters
    ----------
    groups : ndarray, (nobs, nre) or (nobs,)
        array of group/category observations
    sigma : ndarray, (nre+1,)
        array of standard deviations of random effects,
        last element is the standard deviation of the
        idiosyncratic error

    Returns
    -------
    omega : ndarray, (nobs, nobs)
        covariance matrix of error
    omegainv : ndarray, (nobs, nobs)
        inverse covariance matrix of error
    omegainvsqrt : ndarray, (nobs, nobs)
        squareroot inverse covariance matrix of error
        such that omega = omegainvsqrt * omegainvsqrt.T

    Notes
    -----
    This does not use sparse matrices and constructs nobs by nobs
    matrices. Also, omegainvsqrt is not sparse, i.e. elements are non-zero
    """
    if groups.ndim == 1:
        groups = groups[:, None]
    nobs, nre = groups.shape
    omega = sigmas[-1] * np.eye(nobs)
    for igr in range(nre):
        group = groups[:, igr:igr + 1]
        groupuniq = np.unique(group)
        dummygr = sigmas[igr] * (group == groupuniq).astype(float)
        omega += np.dot(dummygr, dummygr.T)
    ev, evec = np.linalg.eigh(omega)
    omegainv = np.dot(evec, (1 / ev * evec).T)
    omegainvhalf = evec / np.sqrt(ev)
    return (omega, omegainv, omegainvhalf)