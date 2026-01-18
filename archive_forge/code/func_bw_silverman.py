import numpy as np
from scipy.stats import scoreatpercentile
from statsmodels.compat.pandas import Substitution
from statsmodels.sandbox.nonparametric import kernels
def bw_silverman(x, kernel=None):
    """
    Silverman's Rule of Thumb

    Parameters
    ----------
    x : array_like
        Array for which to get the bandwidth
    kernel : CustomKernel object
        Unused

    Returns
    -------
    bw : float
        The estimate of the bandwidth

    Notes
    -----
    Returns .9 * A * n ** (-1/5.) where ::

       A = min(std(x, ddof=1), IQR/1.349)
       IQR = np.subtract.reduce(np.percentile(x, [75,25]))

    References
    ----------

    Silverman, B.W. (1986) `Density Estimation.`
    """
    A = _select_sigma(x)
    n = len(x)
    return 0.9 * A * n ** (-0.2)