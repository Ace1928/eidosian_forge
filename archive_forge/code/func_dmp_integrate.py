from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.polyerrors import (
from sympy.utilities import variations
from math import ceil as _ceil, log as _log
def dmp_integrate(f, m, u, K):
    """
    Computes the indefinite integral of ``f`` in ``x_0`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x,y = ring("x,y", QQ)

    >>> R.dmp_integrate(x + 2*y, 1)
    1/2*x**2 + 2*x*y
    >>> R.dmp_integrate(x + 2*y, 2)
    1/6*x**3 + x**2*y

    """
    if not u:
        return dup_integrate(f, m, K)
    if m <= 0 or dmp_zero_p(f, u):
        return f
    g, v = (dmp_zeros(m, u - 1, K), u - 1)
    for i, c in enumerate(reversed(f)):
        n = i + 1
        for j in range(1, m):
            n *= i + j + 1
        g.insert(0, dmp_quo_ground(c, K(n), v, K))
    return g