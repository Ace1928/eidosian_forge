import math
import cupy
from cupy._core import internal  # NOQA
from cupy._core._scalar import get_typename  # NOQA
from cupy_backends.cuda.api import runtime
from cupyx.scipy import special as spec
from cupyx.scipy.interpolate._bspline import BSpline, _get_dtype
import numpy as np
@classmethod
def from_derivatives(cls, xi, yi, orders=None, extrapolate=None):
    """
        Construct a piecewise polynomial in the Bernstein basis,
        compatible with the specified values and derivatives at breakpoints.

        Parameters
        ----------
        xi : array_like
            sorted 1-D array of x-coordinates
        yi : array_like or list of array_likes
            ``yi[i][j]`` is the ``j`` th derivative known at ``xi[i]``
        orders : None or int or array_like of ints. Default: None.
            Specifies the degree of local polynomials. If not None, some
            derivatives are ignored.
        extrapolate : bool or 'periodic', optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used. Default is True.

        Notes
        -----
        If ``k`` derivatives are specified at a breakpoint ``x``, the
        constructed polynomial is exactly ``k`` times continuously
        differentiable at ``x``, unless the ``order`` is provided explicitly.
        In the latter case, the smoothness of the polynomial at
        the breakpoint is controlled by the ``order``.

        Deduces the number of derivatives to match at each end
        from ``order`` and the number of derivatives available. If
        possible it uses the same number of derivatives from
        each end; if the number is odd it tries to take the
        extra one from y2. In any case if not enough derivatives
        are available at one end or another it draws enough to
        make up the total from the other end.

        If the order is too high and not enough derivatives are available,
        an exception is raised.

        Examples
        --------
        >>> from cupyx.scipy.interpolate import BPoly
        >>> BPoly.from_derivatives([0, 1], [[1, 2], [3, 4]])

        Creates a polynomial `f(x)` of degree 3, defined on `[0, 1]`
        such that `f(0) = 1, df/dx(0) = 2, f(1) = 3, df/dx(1) = 4`

        >>> BPoly.from_derivatives([0, 1, 2], [[0, 1], [0], [2]])

        Creates a piecewise polynomial `f(x)`, such that
        `f(0) = f(1) = 0`, `f(2) = 2`, and `df/dx(0) = 1`.
        Based on the number of derivatives provided, the order of the
        local polynomials is 2 on `[0, 1]` and 1 on `[1, 2]`.
        Notice that no restriction is imposed on the derivatives at
        ``x = 1`` and ``x = 2``.

        Indeed, the explicit form of the polynomial is::

            f(x) = | x * (1 - x),  0 <= x < 1
                   | 2 * (x - 1),  1 <= x <= 2

        So that f'(1-0) = -1 and f'(1+0) = 2
        """
    xi = cupy.asarray(xi)
    if len(xi) != len(yi):
        raise ValueError('xi and yi need to have the same length')
    if cupy.any(xi[1:] - xi[:1] <= 0):
        raise ValueError('x coordinates are not in increasing order')
    m = len(xi) - 1
    try:
        k = max((len(yi[i]) + len(yi[i + 1]) for i in range(m)))
    except TypeError as e:
        raise ValueError('Using a 1-D array for y? Please .reshape(-1, 1).') from e
    if orders is None:
        orders = [None] * m
    else:
        if isinstance(orders, (int, cupy.integer)):
            orders = [orders] * m
        k = max(k, max(orders))
        if any((o <= 0 for o in orders)):
            raise ValueError('Orders must be positive.')
    c = []
    for i in range(m):
        y1, y2 = (yi[i], yi[i + 1])
        if orders[i] is None:
            n1, n2 = (len(y1), len(y2))
        else:
            n = orders[i] + 1
            n1 = min(n // 2, len(y1))
            n2 = min(n - n1, len(y2))
            n1 = min(n - n2, len(y2))
            if n1 + n2 != n:
                mesg = 'Point %g has %d derivatives, point %g has %d derivatives, but order %d requested' % (xi[i], len(y1), xi[i + 1], len(y2), orders[i])
                raise ValueError(mesg)
            if not (n1 <= len(y1) and n2 <= len(y2)):
                raise ValueError('`order` input incompatible with length y1 or y2.')
        b = BPoly._construct_from_derivatives(xi[i], xi[i + 1], y1[:n1], y2[:n2])
        if len(b) < k:
            b = BPoly._raise_degree(b, k - len(b))
        c.append(b)
    c = cupy.asarray(c)
    return cls(c.swapaxes(0, 1), xi, extrapolate)