from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Any, cast
import numpy as np
import numpy.typing as npt
import math
import warnings
from collections import namedtuple
from scipy.special import roots_legendre
from scipy.special import gammaln, logsumexp
from scipy._lib._util import _rng_spawn
from scipy._lib.deprecation import (_NoValue, _deprecate_positional_args,
@_deprecate_positional_args(version='1.14')
def simpson(y, *, x=None, dx=1.0, axis=-1, even=_NoValue):
    """
    Integrate y(x) using samples along the given axis and the composite
    Simpson's rule. If x is None, spacing of dx is assumed.

    If there are an even number of samples, N, then there are an odd
    number of intervals (N-1), but Simpson's rule requires an even number
    of intervals. The parameter 'even' controls how this is handled.

    Parameters
    ----------
    y : array_like
        Array to be integrated.
    x : array_like, optional
        If given, the points at which `y` is sampled.
    dx : float, optional
        Spacing of integration points along axis of `x`. Only used when
        `x` is None. Default is 1.
    axis : int, optional
        Axis along which to integrate. Default is the last axis.
    even : {None, 'simpson', 'avg', 'first', 'last'}, optional
        'avg' : Average two results:
            1) use the first N-2 intervals with
               a trapezoidal rule on the last interval and
            2) use the last
               N-2 intervals with a trapezoidal rule on the first interval.

        'first' : Use Simpson's rule for the first N-2 intervals with
                a trapezoidal rule on the last interval.

        'last' : Use Simpson's rule for the last N-2 intervals with a
               trapezoidal rule on the first interval.

        None : equivalent to 'simpson' (default)

        'simpson' : Use Simpson's rule for the first N-2 intervals with the
                  addition of a 3-point parabolic segment for the last
                  interval using equations outlined by Cartwright [1]_.
                  If the axis to be integrated over only has two points then
                  the integration falls back to a trapezoidal integration.

                  .. versionadded:: 1.11.0

        .. versionchanged:: 1.11.0
            The newly added 'simpson' option is now the default as it is more
            accurate in most situations.

        .. deprecated:: 1.11.0
            Parameter `even` is deprecated and will be removed in SciPy
            1.14.0. After this time the behaviour for an even number of
            points will follow that of `even='simpson'`.

    Returns
    -------
    float
        The estimated integral computed with the composite Simpson's rule.

    See Also
    --------
    quad : adaptive quadrature using QUADPACK
    romberg : adaptive Romberg quadrature
    quadrature : adaptive Gaussian quadrature
    fixed_quad : fixed-order Gaussian quadrature
    dblquad : double integrals
    tplquad : triple integrals
    romb : integrators for sampled data
    cumulative_trapezoid : cumulative integration for sampled data
    cumulative_simpson : cumulative integration using Simpson's 1/3 rule
    ode : ODE integrators
    odeint : ODE integrators

    Notes
    -----
    For an odd number of samples that are equally spaced the result is
    exact if the function is a polynomial of order 3 or less. If
    the samples are not equally spaced, then the result is exact only
    if the function is a polynomial of order 2 or less.

    References
    ----------
    .. [1] Cartwright, Kenneth V. Simpson's Rule Cumulative Integration with
           MS Excel and Irregularly-spaced Data. Journal of Mathematical
           Sciences and Mathematics Education. 12 (2): 1-9

    Examples
    --------
    >>> from scipy import integrate
    >>> import numpy as np
    >>> x = np.arange(0, 10)
    >>> y = np.arange(0, 10)

    >>> integrate.simpson(y, x)
    40.5

    >>> y = np.power(x, 3)
    >>> integrate.simpson(y, x)
    1640.5
    >>> integrate.quad(lambda x: x**3, 0, 9)[0]
    1640.25

    >>> integrate.simpson(y, x, even='first')
    1644.5

    """
    y = np.asarray(y)
    nd = len(y.shape)
    N = y.shape[axis]
    last_dx = dx
    first_dx = dx
    returnshape = 0
    if x is not None:
        x = np.asarray(x)
        if len(x.shape) == 1:
            shapex = [1] * nd
            shapex[axis] = x.shape[0]
            saveshape = x.shape
            returnshape = 1
            x = x.reshape(tuple(shapex))
        elif len(x.shape) != len(y.shape):
            raise ValueError('If given, shape of x must be 1-D or the same as y.')
        if x.shape[axis] != N:
            raise ValueError('If given, length of x along axis must be the same as y.')
    if even is not _NoValue:
        warnings.warn("The 'even' keyword is deprecated as of SciPy 1.11.0 and will be removed in SciPy 1.14.0", DeprecationWarning, stacklevel=2)
    if N % 2 == 0:
        val = 0.0
        result = 0.0
        slice_all = (slice(None),) * nd
        even = even if even not in (_NoValue, None) else 'simpson'
        if even not in ['avg', 'last', 'first', 'simpson']:
            raise ValueError("Parameter 'even' must be 'simpson', 'avg', 'last', or 'first'.")
        if N == 2:
            slice1 = tupleset(slice_all, axis, -1)
            slice2 = tupleset(slice_all, axis, -2)
            if x is not None:
                last_dx = x[slice1] - x[slice2]
            val += 0.5 * last_dx * (y[slice1] + y[slice2])
            even = None
        if even == 'simpson':
            result = _basic_simpson(y, 0, N - 3, x, dx, axis)
            slice1 = tupleset(slice_all, axis, -1)
            slice2 = tupleset(slice_all, axis, -2)
            slice3 = tupleset(slice_all, axis, -3)
            h = np.asarray([dx, dx], dtype=np.float64)
            if x is not None:
                hm2 = tupleset(slice_all, axis, slice(-2, -1, 1))
                hm1 = tupleset(slice_all, axis, slice(-1, None, 1))
                diffs = np.float64(np.diff(x, axis=axis))
                h = [np.squeeze(diffs[hm2], axis=axis), np.squeeze(diffs[hm1], axis=axis)]
            num = 2 * h[1] ** 2 + 3 * h[0] * h[1]
            den = 6 * (h[1] + h[0])
            alpha = np.true_divide(num, den, out=np.zeros_like(den), where=den != 0)
            num = h[1] ** 2 + 3.0 * h[0] * h[1]
            den = 6 * h[0]
            beta = np.true_divide(num, den, out=np.zeros_like(den), where=den != 0)
            num = 1 * h[1] ** 3
            den = 6 * h[0] * (h[0] + h[1])
            eta = np.true_divide(num, den, out=np.zeros_like(den), where=den != 0)
            result += alpha * y[slice1] + beta * y[slice2] - eta * y[slice3]
        if even in ['avg', 'first']:
            slice1 = tupleset(slice_all, axis, -1)
            slice2 = tupleset(slice_all, axis, -2)
            if x is not None:
                last_dx = x[slice1] - x[slice2]
            val += 0.5 * last_dx * (y[slice1] + y[slice2])
            result = _basic_simpson(y, 0, N - 3, x, dx, axis)
        if even in ['avg', 'last']:
            slice1 = tupleset(slice_all, axis, 0)
            slice2 = tupleset(slice_all, axis, 1)
            if x is not None:
                first_dx = x[tuple(slice2)] - x[tuple(slice1)]
            val += 0.5 * first_dx * (y[slice2] + y[slice1])
            result += _basic_simpson(y, 1, N - 2, x, dx, axis)
        if even == 'avg':
            val /= 2.0
            result /= 2.0
        result = result + val
    else:
        result = _basic_simpson(y, 0, N - 2, x, dx, axis)
    if returnshape:
        x = x.reshape(saveshape)
    return result