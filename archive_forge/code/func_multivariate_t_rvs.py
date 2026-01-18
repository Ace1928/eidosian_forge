import numpy as np
from scipy import integrate, stats, special
from scipy.stats import chi
from .extras import mvstdnormcdf
from numpy import exp as np_exp
from numpy import log as np_log
from scipy.special import gamma as sps_gamma
from scipy.special import gammaln as sps_gammaln
def multivariate_t_rvs(m, S, df=np.inf, n=1):
    """generate random variables of multivariate t distribution

    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))

    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable


    """
    m = np.asarray(m)
    d = len(m)
    if df == np.inf:
        x = np.ones(n)
    else:
        x = np.random.chisquare(df, n) / df
    z = np.random.multivariate_normal(np.zeros(d), S, (n,))
    return m + z / np.sqrt(x)[:, None]