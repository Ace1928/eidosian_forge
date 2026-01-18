import numpy as np
from scipy.special import erf
def aitchison_aitken(h, Xi, x, num_levels=None):
    """
    The Aitchison-Aitken kernel, used for unordered discrete random variables.

    Parameters
    ----------
    h : 1-D ndarray, shape (K,)
        The bandwidths used to estimate the value of the kernel function.
    Xi : 2-D ndarray of ints, shape (nobs, K)
        The value of the training set.
    x : 1-D ndarray, shape (K,)
        The value at which the kernel density is being estimated.
    num_levels : bool, optional
        Gives the user the option to specify the number of levels for the
        random variable.  If False, the number of levels is calculated from
        the data.

    Returns
    -------
    kernel_value : ndarray, shape (nobs, K)
        The value of the kernel function at each training point for each var.

    Notes
    -----
    See p.18 of [2]_ for details.  The value of the kernel L if :math:`X_{i}=x`
    is :math:`1-\\lambda`, otherwise it is :math:`\\frac{\\lambda}{c-1}`.
    Here :math:`c` is the number of levels plus one of the RV.

    References
    ----------
    .. [*] J. Aitchison and C.G.G. Aitken, "Multivariate binary discrimination
           by the kernel method", Biometrika, vol. 63, pp. 413-420, 1976.
    .. [*] Racine, Jeff. "Nonparametric Econometrics: A Primer," Foundation
           and Trends in Econometrics: Vol 3: No 1, pp1-88., 2008.
    """
    Xi = Xi.reshape(Xi.size)
    if num_levels is None:
        num_levels = np.asarray(np.unique(Xi).size)
    kernel_value = np.ones(Xi.size) * h / (num_levels - 1)
    idx = Xi == x
    kernel_value[idx] = (idx * (1 - h))[idx]
    return kernel_value