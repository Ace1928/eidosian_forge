from __future__ import annotations
import math
import warnings
from collections import namedtuple
import numpy as np
from numpy import (isscalar, r_, log, around, unique, asarray, zeros,
from scipy import optimize, special, interpolate, stats
from scipy._lib._bunch import _make_tuple_bunch
from scipy._lib._util import _rename_parameter, _contains_nan, _get_nan
from ._ansari_swilk_statistics import gscale, swilk
from . import _stats_py
from ._fit import FitResult
from ._stats_py import find_repeats, _normtest_finish, SignificanceResult
from .contingency import chi2_contingency
from . import distributions
from ._distn_infrastructure import rv_generic
from ._hypotests import _get_wilcoxon_distr
from ._axis_nan_policy import _axis_nan_policy_factory
def boxcox(x, lmbda=None, alpha=None, optimizer=None):
    """Return a dataset transformed by a Box-Cox power transformation.

    Parameters
    ----------
    x : ndarray
        Input array to be transformed.

        If `lmbda` is not None, this is an alias of
        `scipy.special.boxcox`.
        Returns nan if ``x < 0``; returns -inf if ``x == 0 and lmbda < 0``.

        If `lmbda` is None, array must be positive, 1-dimensional, and
        non-constant.

    lmbda : scalar, optional
        If `lmbda` is None (default), find the value of `lmbda` that maximizes
        the log-likelihood function and return it as the second output
        argument.

        If `lmbda` is not None, do the transformation for that value.

    alpha : float, optional
        If `lmbda` is None and `alpha` is not None (default), return the
        ``100 * (1-alpha)%`` confidence  interval for `lmbda` as the third
        output argument. Must be between 0.0 and 1.0.

        If `lmbda` is not None, `alpha` is ignored.
    optimizer : callable, optional
        If `lmbda` is None, `optimizer` is the scalar optimizer used to find
        the value of `lmbda` that minimizes the negative log-likelihood
        function. `optimizer` is a callable that accepts one argument:

        fun : callable
            The objective function, which evaluates the negative
            log-likelihood function at a provided value of `lmbda`

        and returns an object, such as an instance of
        `scipy.optimize.OptimizeResult`, which holds the optimal value of
        `lmbda` in an attribute `x`.

        See the example in `boxcox_normmax` or the documentation of
        `scipy.optimize.minimize_scalar` for more information.

        If `lmbda` is not None, `optimizer` is ignored.

    Returns
    -------
    boxcox : ndarray
        Box-Cox power transformed array.
    maxlog : float, optional
        If the `lmbda` parameter is None, the second returned argument is
        the `lmbda` that maximizes the log-likelihood function.
    (min_ci, max_ci) : tuple of float, optional
        If `lmbda` parameter is None and `alpha` is not None, this returned
        tuple of floats represents the minimum and maximum confidence limits
        given `alpha`.

    See Also
    --------
    probplot, boxcox_normplot, boxcox_normmax, boxcox_llf

    Notes
    -----
    The Box-Cox transform is given by::

        y = (x**lmbda - 1) / lmbda,  for lmbda != 0
            log(x),                  for lmbda = 0

    `boxcox` requires the input data to be positive.  Sometimes a Box-Cox
    transformation provides a shift parameter to achieve this; `boxcox` does
    not.  Such a shift parameter is equivalent to adding a positive constant to
    `x` before calling `boxcox`.

    The confidence limits returned when `alpha` is provided give the interval
    where:

    .. math::

        llf(\\hat{\\lambda}) - llf(\\lambda) < \\frac{1}{2}\\chi^2(1 - \\alpha, 1),

    with ``llf`` the log-likelihood function and :math:`\\chi^2` the chi-squared
    function.

    References
    ----------
    G.E.P. Box and D.R. Cox, "An Analysis of Transformations", Journal of the
    Royal Statistical Society B, 26, 211-252 (1964).

    Examples
    --------
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt

    We generate some random variates from a non-normal distribution and make a
    probability plot for it, to show it is non-normal in the tails:

    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(211)
    >>> x = stats.loggamma.rvs(5, size=500) + 5
    >>> prob = stats.probplot(x, dist=stats.norm, plot=ax1)
    >>> ax1.set_xlabel('')
    >>> ax1.set_title('Probplot against normal distribution')

    We now use `boxcox` to transform the data so it's closest to normal:

    >>> ax2 = fig.add_subplot(212)
    >>> xt, _ = stats.boxcox(x)
    >>> prob = stats.probplot(xt, dist=stats.norm, plot=ax2)
    >>> ax2.set_title('Probplot after Box-Cox transformation')

    >>> plt.show()

    """
    x = np.asarray(x)
    if lmbda is not None:
        return special.boxcox(x, lmbda)
    if x.ndim != 1:
        raise ValueError('Data must be 1-dimensional.')
    if x.size == 0:
        return x
    if np.all(x == x[0]):
        raise ValueError('Data must not be constant.')
    if np.any(x <= 0):
        raise ValueError('Data must be positive.')
    lmax = boxcox_normmax(x, method='mle', optimizer=optimizer)
    y = boxcox(x, lmax)
    if alpha is None:
        return (y, lmax)
    else:
        interval = _boxcox_conf_interval(x, lmax, alpha)
        return (y, lmax, interval)