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
def boxcox_normmax(x, brack=None, method='pearsonr', optimizer=None):
    """Compute optimal Box-Cox transform parameter for input data.

    Parameters
    ----------
    x : array_like
        Input array. All entries must be positive, finite, real numbers.
    brack : 2-tuple, optional, default (-2.0, 2.0)
         The starting interval for a downhill bracket search for the default
         `optimize.brent` solver. Note that this is in most cases not
         critical; the final result is allowed to be outside this bracket.
         If `optimizer` is passed, `brack` must be None.
    method : str, optional
        The method to determine the optimal transform parameter (`boxcox`
        ``lmbda`` parameter). Options are:

        'pearsonr'  (default)
            Maximizes the Pearson correlation coefficient between
            ``y = boxcox(x)`` and the expected values for ``y`` if `x` would be
            normally-distributed.

        'mle'
            Maximizes the log-likelihood `boxcox_llf`.  This is the method used
            in `boxcox`.

        'all'
            Use all optimization methods available, and return all results.
            Useful to compare different methods.
    optimizer : callable, optional
        `optimizer` is a callable that accepts one argument:

        fun : callable
            The objective function to be minimized. `fun` accepts one argument,
            the Box-Cox transform parameter `lmbda`, and returns the value of
            the function (e.g., the negative log-likelihood) at the provided
            argument. The job of `optimizer` is to find the value of `lmbda`
            that *minimizes* `fun`.

        and returns an object, such as an instance of
        `scipy.optimize.OptimizeResult`, which holds the optimal value of
        `lmbda` in an attribute `x`.

        See the example below or the documentation of
        `scipy.optimize.minimize_scalar` for more information.

    Returns
    -------
    maxlog : float or ndarray
        The optimal transform parameter found.  An array instead of a scalar
        for ``method='all'``.

    See Also
    --------
    boxcox, boxcox_llf, boxcox_normplot, scipy.optimize.minimize_scalar

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt

    We can generate some data and determine the optimal ``lmbda`` in various
    ways:

    >>> rng = np.random.default_rng()
    >>> x = stats.loggamma.rvs(5, size=30, random_state=rng) + 5
    >>> y, lmax_mle = stats.boxcox(x)
    >>> lmax_pearsonr = stats.boxcox_normmax(x)

    >>> lmax_mle
    2.217563431465757
    >>> lmax_pearsonr
    2.238318660200961
    >>> stats.boxcox_normmax(x, method='all')
    array([2.23831866, 2.21756343])

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> prob = stats.boxcox_normplot(x, -10, 10, plot=ax)
    >>> ax.axvline(lmax_mle, color='r')
    >>> ax.axvline(lmax_pearsonr, color='g', ls='--')

    >>> plt.show()

    Alternatively, we can define our own `optimizer` function. Suppose we
    are only interested in values of `lmbda` on the interval [6, 7], we
    want to use `scipy.optimize.minimize_scalar` with ``method='bounded'``,
    and we want to use tighter tolerances when optimizing the log-likelihood
    function. To do this, we define a function that accepts positional argument
    `fun` and uses `scipy.optimize.minimize_scalar` to minimize `fun` subject
    to the provided bounds and tolerances:

    >>> from scipy import optimize
    >>> options = {'xatol': 1e-12}  # absolute tolerance on `x`
    >>> def optimizer(fun):
    ...     return optimize.minimize_scalar(fun, bounds=(6, 7),
    ...                                     method="bounded", options=options)
    >>> stats.boxcox_normmax(x, optimizer=optimizer)
    6.000...
    """
    if optimizer is None:
        if brack is None:
            brack = (-2.0, 2.0)

        def _optimizer(func, args):
            return optimize.brent(func, args=args, brack=brack)
    else:
        if not callable(optimizer):
            raise ValueError('`optimizer` must be a callable')
        if brack is not None:
            raise ValueError('`brack` must be None if `optimizer` is given')

        def _optimizer(func, args):

            def func_wrapped(x):
                return func(x, *args)
            return getattr(optimizer(func_wrapped), 'x', None)

    def _pearsonr(x):
        osm_uniform = _calc_uniform_order_statistic_medians(len(x))
        xvals = distributions.norm.ppf(osm_uniform)

        def _eval_pearsonr(lmbda, xvals, samps):
            y = boxcox(samps, lmbda)
            yvals = np.sort(y)
            r, prob = _stats_py.pearsonr(xvals, yvals)
            return 1 - r
        return _optimizer(_eval_pearsonr, args=(xvals, x))

    def _mle(x):

        def _eval_mle(lmb, data):
            return -boxcox_llf(lmb, data)
        return _optimizer(_eval_mle, args=(x,))

    def _all(x):
        maxlog = np.empty(2, dtype=float)
        maxlog[0] = _pearsonr(x)
        maxlog[1] = _mle(x)
        return maxlog
    methods = {'pearsonr': _pearsonr, 'mle': _mle, 'all': _all}
    if method not in methods.keys():
        raise ValueError('Method %s not recognized.' % method)
    optimfunc = methods[method]
    try:
        res = optimfunc(x)
    except ValueError as e:
        if 'infs or NaNs' in str(e):
            message = 'The `x` argument of `boxcox_normmax` must contain only positive, finite, real numbers.'
            raise ValueError(message) from e
        else:
            raise e
    if res is None:
        message = 'The `optimizer` argument of `boxcox_normmax` must return an object containing the optimal `lmbda` in attribute `x`.'
        raise ValueError(message)
    else:
        x = np.asarray(x)
        x_treme = np.max(x, axis=0) if np.any(res > 0) else np.min(x, axis=0)
        istransinf = np.isinf(special.boxcox(x_treme, res))
        dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64
        if np.any(istransinf):
            warnings.warn(f'The optimal lambda is {res}, but the returned lambda is theconstrained optimum to ensure that the maximum or the minimum of the transformed data does not cause overflow in {dtype}.', stacklevel=2)
            ymax = np.finfo(dtype).max / 10000
            constrained_res = _boxcox_inv_lmbda(x_treme, ymax * np.sign(res))
            if isinstance(res, np.ndarray):
                res[istransinf] = constrained_res
            else:
                res = constrained_res
    return res