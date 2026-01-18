import numpy as np
from scipy import stats
from statsmodels.tools.rng_qrng import check_random_state
from statsmodels.distributions.copula.copulas import Copula
def rvs_kernel(sample, size, bw=1, k_func=None, return_extras=False):
    """Random sampling from empirical copula using Beta distribution

    Parameters
    ----------
    sample : ndarray
        Sample of multivariate observations in (o, 1) interval.
    size : int
        Number of observations to simulate.
    bw : float
        Bandwidth for Beta sampling. The beta copula corresponds to a kernel
        estimate of the distribution. bw=1 corresponds to the empirical beta
        copula. A small bandwidth like bw=0.001 corresponds to small noise
        added to the empirical distribution. Larger bw, e.g. bw=10 corresponds
        to kernel estimate with more smoothing.
    k_func : None or callable
        The default kernel function is currently a beta function with 1 added
        to the first beta parameter.
    return_extras : bool
        If this is False, then only the random sample will be returned.
        If true, then extra information is returned that is mainly of interest
        for verification.

    Returns
    -------
    rvs : ndarray
        Multivariate sample with ``size`` observations drawn from the Beta
        Copula.

    Notes
    -----
    Status: experimental, API will change.
    """
    n = sample.shape[0]
    if k_func is None:
        kfunc = _kernel_rvs_beta1
    idx = np.random.randint(0, n, size=size)
    xi = sample[idx]
    krvs = np.column_stack([kfunc(xii, bw) for xii in xi.T])
    if return_extras:
        return (krvs, idx, xi)
    else:
        return krvs