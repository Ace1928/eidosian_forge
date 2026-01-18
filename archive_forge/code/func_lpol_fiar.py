from statsmodels.compat.numpy import NP_LT_2
from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from scipy import linalg, optimize, signal
from statsmodels.tools.docstring import Docstring, remove_parameters
from statsmodels.tools.validation import array_like
def lpol_fiar(d, n=20):
    """AR representation of fractional integration

    .. math:: (1-L)^{d} for |d|<0.5  or |d|<1 (?)

    Parameters
    ----------
    d : float
        fractional power
    n : int
        number of terms to calculate, including lag zero

    Returns
    -------
    ar : ndarray
        coefficients of lag polynomial

    Notes:
    first coefficient is 1, negative signs except for first term,
    ar(L)*x_t
    """
    from scipy.special import gammaln
    j = np.arange(n)
    ar = -np.exp(gammaln(-d + j) - gammaln(j + 1) - gammaln(-d))
    ar[0] = 1
    return ar