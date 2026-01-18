from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.polyerrors import (
from sympy.utilities import variations
from math import ceil as _ceil, log as _log
def dmp_eval(f, a, u, K):
    """
    Evaluate a polynomial at ``x_0 = a`` in ``K[X]`` using the Horner scheme.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_eval(2*x*y + 3*x + y + 2, 2)
    5*y + 8

    """
    if not u:
        return dup_eval(f, a, K)
    if not a:
        return dmp_TC(f, K)
    result, v = (dmp_LC(f, K), u - 1)
    for coeff in f[1:]:
        result = dmp_mul_ground(result, a, v, K)
        result = dmp_add(result, coeff, v, K)
    return result