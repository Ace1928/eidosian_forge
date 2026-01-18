from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.polyerrors import (
from sympy.utilities import variations
from math import ceil as _ceil, log as _log
def dmp_eval_tail(f, A, u, K):
    """
    Evaluate a polynomial at ``x_j = a_j, ...`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = 2*x*y + 3*x + y + 2

    >>> R.dmp_eval_tail(f, [2])
    7*x + 4
    >>> R.dmp_eval_tail(f, [2, 2])
    18

    """
    if not A:
        return f
    if dmp_zero_p(f, u):
        return dmp_zero(u - len(A))
    e = _rec_eval_tail(f, 0, A, u, K)
    if u == len(A) - 1:
        return e
    else:
        return dmp_strip(e, u - len(A))