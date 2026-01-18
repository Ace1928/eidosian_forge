import numpy as np
from scipy.stats import norm as Gaussian
from statsmodels.tools import tools
from statsmodels.tools.validation import array_like, float_like
from . import norms
from ._qn import _qn
def qn_scale(a, c=1 / (np.sqrt(2) * Gaussian.ppf(5 / 8)), axis=0):
    """
    Computes the Qn robust estimator of scale

    The Qn scale estimator is a more efficient alternative to the MAD.
    The Qn scale estimator of an array a of length n is defined as
    c * {abs(a[i] - a[j]): i<j}_(k), for k equal to [n/2] + 1 choose 2. Thus,
    the Qn estimator is the k-th order statistic of the absolute differences
    of the array. The optional constant is used to normalize the estimate
    as explained below. The implementation follows the algorithm described
    in Croux and Rousseeuw (1992).

    Parameters
    ----------
    a : array_like
        Input array.
    c : float, optional
        The normalization constant. The default value is used to get consistent
        estimates of the standard deviation at the normal distribution.
    axis : int, optional
        The default is 0.

    Returns
    -------
    {float, ndarray}
        The Qn robust estimator of scale
    """
    a = array_like(a, 'a', ndim=None, dtype=np.float64, contiguous=True, order='C')
    c = float_like(c, 'c')
    if a.ndim == 0:
        raise ValueError('a should have at least one dimension')
    elif a.size == 0:
        return np.nan
    else:
        out = np.apply_along_axis(_qn, axis=axis, arr=a, c=c)
        if out.ndim == 0:
            return float(out)
        return out