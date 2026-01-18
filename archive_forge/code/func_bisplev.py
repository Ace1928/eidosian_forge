import warnings
import numpy as np
from . import _fitpack
from numpy import (atleast_1d, array, ones, zeros, sqrt, ravel, transpose,
from . import dfitpack
def bisplev(x, y, tck, dx=0, dy=0):
    """
    Evaluate a bivariate B-spline and its derivatives.

    Return a rank-2 array of spline function values (or spline derivative
    values) at points given by the cross-product of the rank-1 arrays `x` and
    `y`.  In special cases, return an array or just a float if either `x` or
    `y` or both are floats.  Based on BISPEV and PARDER from FITPACK.

    Parameters
    ----------
    x, y : ndarray
        Rank-1 arrays specifying the domain over which to evaluate the
        spline or its derivative.
    tck : tuple
        A sequence of length 5 returned by `bisplrep` containing the knot
        locations, the coefficients, and the degree of the spline:
        [tx, ty, c, kx, ky].
    dx, dy : int, optional
        The orders of the partial derivatives in `x` and `y` respectively.

    Returns
    -------
    vals : ndarray
        The B-spline or its derivative evaluated over the set formed by
        the cross-product of `x` and `y`.

    See Also
    --------
    splprep, splrep, splint, sproot, splev
    UnivariateSpline, BivariateSpline

    Notes
    -----
        See `bisplrep` to generate the `tck` representation.

    References
    ----------
    .. [1] Dierckx P. : An algorithm for surface fitting
       with spline functions
       Ima J. Numer. Anal. 1 (1981) 267-283.
    .. [2] Dierckx P. : An algorithm for surface fitting
       with spline functions
       report tw50, Dept. Computer Science,K.U.Leuven, 1980.
    .. [3] Dierckx P. : Curve and surface fitting with splines,
       Monographs on Numerical Analysis, Oxford University Press, 1993.

    Examples
    --------
    Examples are given :ref:`in the tutorial <tutorial-interpolate_2d_spline>`.

    """
    tx, ty, c, kx, ky = tck
    if not 0 <= dx < kx:
        raise ValueError('0 <= dx = %d < kx = %d must hold' % (dx, kx))
    if not 0 <= dy < ky:
        raise ValueError('0 <= dy = %d < ky = %d must hold' % (dy, ky))
    x, y = map(atleast_1d, [x, y])
    if len(x.shape) != 1 or len(y.shape) != 1:
        raise ValueError('First two entries should be rank-1 arrays.')
    msg = 'Too many data points to interpolate.'
    _int_overflow(x.size * y.size, MemoryError, msg=msg)
    if dx != 0 or dy != 0:
        _int_overflow((tx.size - kx - 1) * (ty.size - ky - 1), MemoryError, msg=msg)
        z, ier = dfitpack.parder(tx, ty, c, kx, ky, dx, dy, x, y)
    else:
        z, ier = dfitpack.bispev(tx, ty, c, kx, ky, x, y)
    if ier == 10:
        raise ValueError('Invalid input data')
    if ier:
        raise TypeError('An error occurred')
    z.shape = (len(x), len(y))
    if len(z) > 1:
        return z
    if len(z[0]) > 1:
        return z[0]
    return z[0][0]