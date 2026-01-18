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
def cumulative_simpson(y, *, x=None, dx=1.0, axis=-1, initial=None):
    """
    Cumulatively integrate y(x) using the composite Simpson's 1/3 rule.
    The integral of the samples at every point is calculated by assuming a 
    quadratic relationship between each point and the two adjacent points.

    Parameters
    ----------
    y : array_like
        Values to integrate. Requires at least one point along `axis`. If two or fewer
        points are provided along `axis`, Simpson's integration is not possible and the
        result is calculated with `cumulative_trapezoid`.
    x : array_like, optional
        The coordinate to integrate along. Must have the same shape as `y` or
        must be 1D with the same length as `y` along `axis`. `x` must also be
        strictly increasing along `axis`.
        If `x` is None (default), integration is performed using spacing `dx`
        between consecutive elements in `y`.
    dx : scalar or array_like, optional
        Spacing between elements of `y`. Only used if `x` is None. Can either 
        be a float, or an array with the same shape as `y`, but of length one along
        `axis`. Default is 1.0.
    axis : int, optional
        Specifies the axis to integrate along. Default is -1 (last axis).
    initial : scalar or array_like, optional
        If given, insert this value at the beginning of the returned result,
        and add it to the rest of the result. Default is None, which means no
        value at ``x[0]`` is returned and `res` has one element less than `y`
        along the axis of integration. Can either be a float, or an array with
        the same shape as `y`, but of length one along `axis`.

    Returns
    -------
    res : ndarray
        The result of cumulative integration of `y` along `axis`.
        If `initial` is None, the shape is such that the axis of integration
        has one less value than `y`. If `initial` is given, the shape is equal
        to that of `y`.

    See Also
    --------
    numpy.cumsum
    cumulative_trapezoid : cumulative integration using the composite 
        trapezoidal rule
    simpson : integrator for sampled data using the Composite Simpson's Rule

    Notes
    -----

    .. versionadded:: 1.12.0

    The composite Simpson's 1/3 method can be used to approximate the definite 
    integral of a sampled input function :math:`y(x)` [1]_. The method assumes 
    a quadratic relationship over the interval containing any three consecutive
    sampled points.

    Consider three consecutive points: 
    :math:`(x_1, y_1), (x_2, y_2), (x_3, y_3)`.

    Assuming a quadratic relationship over the three points, the integral over
    the subinterval between :math:`x_1` and :math:`x_2` is given by formula
    (8) of [2]_:
    
    .. math::
        \\int_{x_1}^{x_2} y(x) dx\\ &= \\frac{x_2-x_1}{6}\\left[\\
        \\left\\{3-\\frac{x_2-x_1}{x_3-x_1}\\right\\} y_1 + \\
        \\left\\{3 + \\frac{(x_2-x_1)^2}{(x_3-x_2)(x_3-x_1)} + \\
        \\frac{x_2-x_1}{x_3-x_1}\\right\\} y_2\\\\
        - \\frac{(x_2-x_1)^2}{(x_3-x_2)(x_3-x_1)} y_3\\right]

    The integral between :math:`x_2` and :math:`x_3` is given by swapping
    appearances of :math:`x_1` and :math:`x_3`. The integral is estimated
    separately for each subinterval and then cumulatively summed to obtain
    the final result.
    
    For samples that are equally spaced, the result is exact if the function
    is a polynomial of order three or less [1]_ and the number of subintervals
    is even. Otherwise, the integral is exact for polynomials of order two or
    less. 

    References
    ----------
    .. [1] Wikipedia page: https://en.wikipedia.org/wiki/Simpson's_rule
    .. [2] Cartwright, Kenneth V. Simpson's Rule Cumulative Integration with
            MS Excel and Irregularly-spaced Data. Journal of Mathematical
            Sciences and Mathematics Education. 12 (2): 1-9

    Examples
    --------
    >>> from scipy import integrate
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-2, 2, num=20)
    >>> y = x**2
    >>> y_int = integrate.cumulative_simpson(y, x=x, initial=0)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, y_int, 'ro', x, x**3/3 - (x[0])**3/3, 'b-')
    >>> ax.grid()
    >>> plt.show()

    The output of `cumulative_simpson` is similar to that of iteratively
    calling `simpson` with successively higher upper limits of integration, but
    not identical.

    >>> def cumulative_simpson_reference(y, x):
    ...     return np.asarray([integrate.simpson(y[:i], x=x[:i])
    ...                        for i in range(2, len(y) + 1)])
    >>>
    >>> rng = np.random.default_rng(354673834679465)
    >>> x, y = rng.random(size=(2, 10))
    >>> x.sort()
    >>>
    >>> res = integrate.cumulative_simpson(y, x=x)
    >>> ref = cumulative_simpson_reference(y, x)
    >>> equal = np.abs(res - ref) < 1e-15
    >>> equal  # not equal when `simpson` has even number of subintervals
    array([False,  True, False,  True, False,  True, False,  True,  True])

    This is expected: because `cumulative_simpson` has access to more
    information than `simpson`, it can typically produce more accurate
    estimates of the underlying integral over subintervals.

    """
    y = _ensure_float_array(y)
    original_y = y
    original_shape = y.shape
    try:
        y = np.swapaxes(y, axis, -1)
    except IndexError as e:
        message = f'`axis={axis}` is not valid for `y` with `y.ndim={y.ndim}`.'
        raise ValueError(message) from e
    if y.shape[-1] < 3:
        res = cumulative_trapezoid(original_y, x, dx=dx, axis=axis, initial=None)
        res = np.swapaxes(res, axis, -1)
    elif x is not None:
        x = _ensure_float_array(x)
        message = 'If given, shape of `x` must be the same as `y` or 1-D with the same length as `y` along `axis`.'
        if not (x.shape == original_shape or (x.ndim == 1 and len(x) == original_shape[axis])):
            raise ValueError(message)
        x = np.broadcast_to(x, y.shape) if x.ndim == 1 else np.swapaxes(x, axis, -1)
        dx = np.diff(x, axis=-1)
        if np.any(dx <= 0):
            raise ValueError('Input x must be strictly increasing.')
        res = _cumulatively_sum_simpson_integrals(y, dx, _cumulative_simpson_unequal_intervals)
    else:
        dx = _ensure_float_array(dx)
        final_dx_shape = tupleset(original_shape, axis, original_shape[axis] - 1)
        alt_input_dx_shape = tupleset(original_shape, axis, 1)
        message = 'If provided, `dx` must either be a scalar or have the same shape as `y` but with only 1 point along `axis`.'
        if not (dx.ndim == 0 or dx.shape == alt_input_dx_shape):
            raise ValueError(message)
        dx = np.broadcast_to(dx, final_dx_shape)
        dx = np.swapaxes(dx, axis, -1)
        res = _cumulatively_sum_simpson_integrals(y, dx, _cumulative_simpson_equal_intervals)
    if initial is not None:
        initial = _ensure_float_array(initial)
        alt_initial_input_shape = tupleset(original_shape, axis, 1)
        message = 'If provided, `initial` must either be a scalar or have the same shape as `y` but with only 1 point along `axis`.'
        if not (initial.ndim == 0 or initial.shape == alt_initial_input_shape):
            raise ValueError(message)
        initial = np.broadcast_to(initial, alt_initial_input_shape)
        initial = np.swapaxes(initial, axis, -1)
        res += initial
        res = np.concatenate((initial, res), axis=-1)
    res = np.swapaxes(res, -1, axis)
    return res