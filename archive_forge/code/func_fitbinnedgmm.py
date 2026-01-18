import numpy as np
from scipy import stats, optimize, special
def fitbinnedgmm(distfn, freq, binedges, start, fixed=None, weightsoptimal=True):
    """estimate parameters of distribution function for binned data using GMM

    Parameters
    ----------
    distfn : distribution instance
        needs to have cdf method, as in scipy.stats
    freq : ndarray, 1d
        frequency count, e.g. obtained by histogram
    binedges : ndarray, 1d
        binedges including lower and upper bound
    start : tuple or array_like ?
        starting values, needs to have correct length
    fixed : None
        not used yet
    weightsoptimal : bool
        If true, then the optimal weighting matrix for GMM is used. If false,
        then the identity matrix is used

    Returns
    -------
    paramest : ndarray
        estimated parameters

    Notes
    -----
    todo: add fixed parameter option

    added factorial

    """
    if fixed is not None:
        raise NotImplementedError
    nobs = np.sum(freq)
    if weightsoptimal:
        weights = freq / float(nobs)
    else:
        weights = np.ones(len(freq))
    freqnormed = freq / float(nobs)

    def gmmobjective(params):
        """negative loglikelihood function of binned data

        corresponds to multinomial
        """
        prob = np.diff(distfn.cdf(binedges, *params))
        momcond = freqnormed - prob
        return np.dot(momcond * weights, momcond)
    return optimize.fmin(gmmobjective, start)