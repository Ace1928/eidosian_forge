from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.polyerrors import (
from sympy.utilities import variations
from math import ceil as _ceil, log as _log
def dmp_diff(f, m, u, K):
    """
    ``m``-th order derivative in ``x_0`` of a polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = x*y**2 + 2*x*y + 3*x + 2*y**2 + 3*y + 1

    >>> R.dmp_diff(f, 1)
    y**2 + 2*y + 3
    >>> R.dmp_diff(f, 2)
    0

    """
    if not u:
        return dup_diff(f, m, K)
    if m <= 0:
        return f
    n = dmp_degree(f, u)
    if n < m:
        return dmp_zero(u)
    deriv, v = ([], u - 1)
    if m == 1:
        for coeff in f[:-m]:
            deriv.append(dmp_mul_ground(coeff, K(n), v, K))
            n -= 1
    else:
        for coeff in f[:-m]:
            k = n
            for i in range(n - 1, n - m, -1):
                k *= i
            deriv.append(dmp_mul_ground(coeff, K(k), v, K))
            n -= 1
    return dmp_strip(deriv, u)