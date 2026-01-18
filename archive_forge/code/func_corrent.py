from statsmodels.compat.python import lzip, lmap
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import logsumexp as sp_logsumexp
def corrent(px, py, pxpy, logbase=2):
    """
    An information theoretic correlation measure.

    Reflects linear and nonlinear correlation between two random variables
    X and Y, characterized by the discrete probability distributions px and py
    respectively.

    Parameters
    ----------
    px : array_like
        Discrete probability distribution of random variable X
    py : array_like
        Discrete probability distribution of random variable Y
    pxpy : 2d array_like, optional
        Joint probability distribution of X and Y.  If pxpy is None, X and Y
        are assumed to be independent.
    logbase : int or np.e, optional
        Default is 2 (bits)

    Returns
    -------
    mutualinfo(px,py,pxpy,logbase=logbase)/shannonentropy(py,logbase=logbase)

    Notes
    -----
    This is also equivalent to

    corrent(px,py,pxpy) = 1 - condent(px,py,pxpy)/shannonentropy(py)
    """
    if not _isproperdist(px) or not _isproperdist(py):
        raise ValueError('px or py is not a proper probability distribution')
    if pxpy is not None and (not _isproperdist(pxpy)):
        raise ValueError('pxpy is not a proper joint distribtion')
    if pxpy is None:
        pxpy = np.outer(py, px)
    return mutualinfo(px, py, pxpy, logbase=logbase) / shannonentropy(py, logbase=logbase)