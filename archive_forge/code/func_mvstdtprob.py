import numpy as np
from scipy import integrate, stats, special
from scipy.stats import chi
from .extras import mvstdnormcdf
from numpy import exp as np_exp
from numpy import log as np_log
from scipy.special import gamma as sps_gamma
from scipy.special import gammaln as sps_gammaln
def mvstdtprob(a, b, R, df, ieps=1e-05, quadkwds=None, mvstkwds=None):
    """
    Probability of rectangular area of standard t distribution

    assumes mean is zero and R is correlation matrix

    Notes
    -----
    This function does not calculate the estimate of the combined error
    between the underlying multivariate normal probability calculations
    and the integration.
    """
    kwds = dict(args=(a, b, R, df), epsabs=0.0001, epsrel=0.01, limit=150)
    if quadkwds is not None:
        kwds.update(quadkwds)
    lower, upper = chi.ppf([ieps, 1 - ieps], df)
    res, err = integrate.quad(funbgh2, lower, upper, **kwds)
    prob = res * bghfactor(df)
    return prob