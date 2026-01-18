from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.polyerrors import (
from sympy.utilities import variations
from math import ceil as _ceil, log as _log
def dmp_eval_in(f, a, j, u, K):
    """
    Evaluate a polynomial at ``x_j = a`` in ``K[X]`` using the Horner scheme.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = 2*x*y + 3*x + y + 2

    >>> R.dmp_eval_in(f, 2, 0)
    5*y + 8
    >>> R.dmp_eval_in(f, 2, 1)
    7*x + 4

    """
    if j < 0 or j > u:
        raise IndexError('0 <= j <= %s expected, got %s' % (u, j))
    return _rec_eval_in(f, a, u, 0, j, K)