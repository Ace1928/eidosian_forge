from statsmodels.compat.python import lmap
import numpy as np
from scipy import stats, optimize, integrate
def distfitbootstrap(sample, distr, nrepl=100):
    """run bootstrap for estimation of distribution parameters

    hard coded: only one shape parameter is allowed and estimated,
        loc=0 and scale=1 are fixed in the estimation

    Parameters
    ----------
    sample : ndarray
        original sample data for bootstrap
    distr : distribution instance with fit_fr method
    nrepl : int
        number of bootstrap replications

    Returns
    -------
    res : array (nrepl,)
        parameter estimates for all bootstrap replications

    """
    nobs = len(sample)
    res = np.zeros(nrepl)
    for ii in range(nrepl):
        rvsind = np.random.randint(nobs, size=nobs)
        x = sample[rvsind]
        res[ii] = distr.fit_fr(x, frozen=[np.nan, 0.0, 1.0])
    return res