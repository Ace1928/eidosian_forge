from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.polyerrors import (
from sympy.utilities import variations
from math import ceil as _ceil, log as _log
def dmp_integrate_in(f, m, j, u, K):
    """
    Computes the indefinite integral of ``f`` in ``x_j`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x,y = ring("x,y", QQ)

    >>> R.dmp_integrate_in(x + 2*y, 1, 0)
    1/2*x**2 + 2*x*y
    >>> R.dmp_integrate_in(x + 2*y, 1, 1)
    x*y + y**2

    """
    if j < 0 or j > u:
        raise IndexError('0 <= j <= u expected, got u = %d, j = %d' % (u, j))
    return _rec_integrate_in(f, m, u, 0, j, K)