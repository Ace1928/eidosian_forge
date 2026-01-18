import numpy as np
from scipy.stats import scoreatpercentile
from statsmodels.compat.pandas import Substitution
from statsmodels.sandbox.nonparametric import kernels
def bw_scott(x, kernel=None):
    """
    Scott's Rule of Thumb

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
    Returns 1.059 * A * n ** (-1/5.) where ::

       A = min(std(x, ddof=1), IQR/1.349)
       IQR = np.subtract.reduce(np.percentile(x, [75,25]))

    References
    ----------

    Scott, D.W. (1992) Multivariate Density Estimation: Theory, Practice, and
        Visualization.
    """
    A = _select_sigma(x)
    n = len(x)
    return 1.059 * A * n ** (-0.2)