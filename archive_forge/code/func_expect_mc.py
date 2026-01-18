import numpy as np
from scipy import special
from statsmodels.sandbox.distributions.multivariate import mvstdtprob
from .extras import mvnormcdf
def expect_mc(dist, func=lambda x: 1, size=50000):
    """calculate expected value of function by Monte Carlo integration

    Parameters
    ----------
    dist : distribution instance
        needs to have rvs defined as a method for drawing random numbers
    func : callable
        function for which expectation is calculated, this function needs to
        be vectorized, integration is over axis=0
    size : int
        number of random samples to use in the Monte Carlo integration,


    Notes
    -----
    this does not batch

    Returns
    -------
    expected value : ndarray
        return of function func integrated over axis=0 by MonteCarlo, this will
        have the same shape as the return of func without axis=0

    Examples
    --------

    integrate probability that both observations are negative

    >>> mvn = mve.MVNormal([0,0],2.)
    >>> mve.expect_mc(mvn, lambda x: (x<np.array([0,0])).all(-1), size=100000)
    0.25306000000000001

    get tail probabilities of marginal distribution (should be 0.1)

    >>> c = stats.norm.isf(0.05, scale=np.sqrt(2.))
    >>> expect_mc(mvn, lambda x: (np.abs(x)>np.array([c, c])), size=100000)
    array([ 0.09969,  0.0986 ])

    or calling the method

    >>> mvn.expect_mc(lambda x: (np.abs(x)>np.array([c, c])), size=100000)
    array([ 0.09937,  0.10075])


    """

    def fun(x):
        return func(x)
    rvs = dist.rvs(size=size)
    return fun(rvs).mean(0)