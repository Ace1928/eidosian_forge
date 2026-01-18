import warnings
import numpy as np
from scipy import stats
from statsmodels.tools.validation import array_like, bool_like, int_like
def anderson_statistic(x, dist='norm', fit=True, params=(), axis=0):
    """
    Calculate the Anderson-Darling a2 statistic.

    Parameters
    ----------
    x : array_like
        The data to test.
    dist : {'norm', callable}
        The assumed distribution under the null of test statistic.
    fit : bool
        If True, then the distribution parameters are estimated.
        Currently only for 1d data x, except in case dist='norm'.
    params : tuple
        The optional distribution parameters if fit is False.
    axis : int
        If dist is 'norm' or fit is False, then data can be an n-dimensional
        and axis specifies the axis of a variable.

    Returns
    -------
    {float, ndarray}
        The Anderson-Darling statistic.
    """
    x = array_like(x, 'x', ndim=None)
    fit = bool_like(fit, 'fit')
    axis = int_like(axis, 'axis')
    y = np.sort(x, axis=axis)
    nobs = y.shape[axis]
    if fit:
        if dist == 'norm':
            xbar = np.expand_dims(np.mean(x, axis=axis), axis)
            s = np.expand_dims(np.std(x, ddof=1, axis=axis), axis)
            w = (y - xbar) / s
            z = stats.norm.cdf(w)
        elif callable(dist):
            params = dist.fit(x)
            z = dist.cdf(y, *params)
        else:
            raise ValueError("dist must be 'norm' or a Callable")
    elif callable(dist):
        z = dist.cdf(y, *params)
    else:
        raise ValueError('if fit is false, then dist must be callable')
    i = np.arange(1, nobs + 1)
    sl1 = [None] * x.ndim
    sl1[axis] = slice(None)
    sl1 = tuple(sl1)
    sl2 = [slice(None)] * x.ndim
    sl2[axis] = slice(None, None, -1)
    sl2 = tuple(sl2)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='divide by zero encountered in log1p')
        ad_values = (2 * i[sl1] - 1.0) / nobs * (np.log(z) + np.log1p(-z[sl2]))
        s = np.sum(ad_values, axis=axis)
    a2 = -nobs - s
    return a2