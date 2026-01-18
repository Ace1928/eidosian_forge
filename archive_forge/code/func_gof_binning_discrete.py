from statsmodels.compat.python import lrange
import numpy as np
from scipy import stats
def gof_binning_discrete(rvs, distfn, arg, nsupp=20):
    """get bins for chisquare type gof tests for a discrete distribution

    Parameters
    ----------
    rvs : ndarray
        sample data
    distname : str
        name of distribution function
    arg : sequence
        parameters of distribution
    nsupp : int
        number of bins. The algorithm tries to find bins with equal weights.
        depending on the distribution, the actual number of bins can be smaller.

    Returns
    -------
    freq : ndarray
        empirical frequencies for sample; not normalized, adds up to sample size
    expfreq : ndarray
        theoretical frequencies according to distribution
    histsupp : ndarray
        bin boundaries for histogram, (added 1e-8 for numerical robustness)

    Notes
    -----
    The results can be used for a chisquare test ::

        (chis,pval) = stats.chisquare(freq, expfreq)

    originally written for scipy.stats test suite,
    still needs to be checked for standalone usage, insufficient input checking
    may not run yet (after copy/paste)

    refactor: maybe a class, check returns, or separate binning from
        test results
    todo :
      optimal number of bins ? (check easyfit),
      recommendation in literature at least 5 expected observations in each bin

    """
    n = len(rvs)
    wsupp = 1.0 / nsupp
    distsupport = lrange(max(distfn.a, -1000), min(distfn.b, 1000) + 1)
    last = 0
    distsupp = [max(distfn.a, -1000)]
    distmass = []
    for ii in distsupport:
        current = distfn.cdf(ii, *arg)
        if current - last >= wsupp - 1e-14:
            distsupp.append(ii)
            distmass.append(current - last)
            last = current
            if current > 1 - wsupp:
                break
    if distsupp[-1] < distfn.b:
        distsupp.append(distfn.b)
        distmass.append(1 - last)
    distsupp = np.array(distsupp)
    distmass = np.array(distmass)
    histsupp = distsupp + 1e-08
    histsupp[0] = distfn.a
    freq, hsupp = np.histogram(rvs, histsupp)
    cdfs = distfn.cdf(distsupp, *arg)
    return (np.array(freq), n * distmass, histsupp)