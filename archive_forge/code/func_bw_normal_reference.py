import numpy as np
from scipy.stats import scoreatpercentile
from statsmodels.compat.pandas import Substitution
from statsmodels.sandbox.nonparametric import kernels
def bw_normal_reference(x, kernel=None):
    """
    Plug-in bandwidth with kernel specific constant based on normal reference.

    This bandwidth minimizes the mean integrated square error if the true
    distribution is the normal. This choice is an appropriate bandwidth for
    single peaked distributions that are similar to the normal distribution.

    Parameters
    ----------
    x : array_like
        Array for which to get the bandwidth
    kernel : CustomKernel object
        Used to calculate the constant for the plug-in bandwidth.
        The default is a Gaussian kernel.

    Returns
    -------
    bw : float
        The estimate of the bandwidth

    Notes
    -----
    Returns C * A * n ** (-1/5.) where ::

       A = min(std(x, ddof=1), IQR/1.349)
       IQR = np.subtract.reduce(np.percentile(x, [75,25]))
       C = constant from Hansen (2009)

    When using a Gaussian kernel this is equivalent to the 'scott' bandwidth up
    to two decimal places. This is the accuracy to which the 'scott' constant is
    specified.

    References
    ----------

    Silverman, B.W. (1986) `Density Estimation.`
    Hansen, B.E. (2009) `Lecture Notes on Nonparametrics.`
    """
    if kernel is None:
        kernel = kernels.Gaussian()
    C = kernel.normal_reference_constant
    A = _select_sigma(x)
    n = len(x)
    return C * A * n ** (-0.2)