from statsmodels.compat.python import lzip, lmap
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import logsumexp as sp_logsumexp
def mutualinfo(px, py, pxpy, logbase=2):
    """
    Returns the mutual information between X and Y.

    Parameters
    ----------
    px : array_like
        Discrete probability distribution of random variable X
    py : array_like
        Discrete probability distribution of random variable Y
    pxpy : 2d array_like
        The joint probability distribution of random variables X and Y.
        Note that if X and Y are independent then the mutual information
        is zero.
    logbase : int or np.e, optional
        Default is 2 (bits)

    Returns
    -------
    shannonentropy(px) - condentropy(px,py,pxpy)
    """
    if not _isproperdist(px) or not _isproperdist(py):
        raise ValueError('px or py is not a proper probability distribution')
    if pxpy is not None and (not _isproperdist(pxpy)):
        raise ValueError('pxpy is not a proper joint distribtion')
    if pxpy is None:
        pxpy = np.outer(py, px)
    return shannonentropy(px, logbase=logbase) - condentropy(px, py, pxpy, logbase=logbase)