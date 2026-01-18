import warnings
from numpy import (logical_and, asarray, pi, zeros_like,
from numpy import (sqrt, exp, greater, less, cos, add, sin, less_equal,
from ._spline import cspline2d, sepfir2d
from ._signaltools import lfilter, sosfilt, lfiltic
from scipy.special import comb
from scipy._lib._util import float_factorial
from scipy.interpolate import BSpline
def cubic(x):
    """
    .. deprecated:: 1.11.0

        `scipy.signal.cubic` is deprecated in SciPy 1.11 and will be
        removed in SciPy 1.13.

        The exact equivalent (for a float array `x`) is::

            >>> from scipy.interpolate import BSpline
            >>> out = BSpline.basis_element([-2, -1, 0, 1, 2])(x)
            >>> out[(x < -2 | (x > 2)] = 0.0

    A cubic B-spline.

    This is a special case of `bspline`, and equivalent to ``bspline(x, 3)``.

    Parameters
    ----------
    x : array_like
        a knot vector

    Returns
    -------
    res : ndarray
        Cubic B-spline basis function values

    See Also
    --------
    bspline : B-spline basis function of order n
    quadratic : A quadratic B-spline.

    Examples
    --------
    We can calculate B-Spline basis function of several orders:

    >>> import numpy as np
    >>> from scipy.signal import bspline, cubic, quadratic
    >>> bspline(0.0, 1)
    1

    >>> knots = [-1.0, 0.0, -1.0]
    >>> bspline(knots, 2)
    array([0.125, 0.75, 0.125])

    >>> np.array_equal(bspline(knots, 2), quadratic(knots))
    True

    >>> np.array_equal(bspline(knots, 3), cubic(knots))
    True

    """
    warnings.warn(msg_cubic, DeprecationWarning, stacklevel=2)
    ax = abs(asarray(x, dtype=float))
    res = zeros_like(ax)
    cond1 = less(ax, 1)
    if cond1.any():
        ax1 = ax[cond1]
        res[cond1] = 2.0 / 3 - 1.0 / 2 * ax1 ** 2 * (2 - ax1)
    cond2 = ~cond1 & less(ax, 2)
    if cond2.any():
        ax2 = ax[cond2]
        res[cond2] = 1.0 / 6 * (2 - ax2) ** 3
    return res