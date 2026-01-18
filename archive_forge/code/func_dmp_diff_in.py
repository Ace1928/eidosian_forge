from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.polyerrors import (
from sympy.utilities import variations
from math import ceil as _ceil, log as _log
def dmp_diff_in(f, m, j, u, K):
    """
    ``m``-th order derivative in ``x_j`` of a polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = x*y**2 + 2*x*y + 3*x + 2*y**2 + 3*y + 1

    >>> R.dmp_diff_in(f, 1, 0)
    y**2 + 2*y + 3
    >>> R.dmp_diff_in(f, 1, 1)
    2*x*y + 2*x + 4*y + 3

    """
    if j < 0 or j > u:
        raise IndexError('0 <= j <= %s expected, got %s' % (u, j))
    return _rec_diff_in(f, m, u, 0, j, K)