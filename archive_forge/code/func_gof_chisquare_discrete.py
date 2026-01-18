from statsmodels.compat.python import lrange
import numpy as np
from scipy import stats
def gof_chisquare_discrete(distfn, arg, rvs, alpha, msg):
    """perform chisquare test for random sample of a discrete distribution

    Parameters
    ----------
    distname : str
        name of distribution function
    arg : sequence
        parameters of distribution
    alpha : float
        significance level, threshold for p-value

    Returns
    -------
    result : bool
        0 if test passes, 1 if test fails

    Notes
    -----
    originally written for scipy.stats test suite,
    still needs to be checked for standalone usage, insufficient input checking
    may not run yet (after copy/paste)

    refactor: maybe a class, check returns, or separate binning from
        test results
    """
    n = len(rvs)
    nsupp = 20
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
    chis, pval = stats.chisquare(np.array(freq), n * distmass)
    return (chis, pval, pval > alpha, 'chisquare - test for %sat arg = %s with pval = %s' % (msg, str(arg), str(pval)))