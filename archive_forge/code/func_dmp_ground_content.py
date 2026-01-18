from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.polyerrors import (
from sympy.utilities import variations
from math import ceil as _ceil, log as _log
def dmp_ground_content(f, u, K):
    """
    Compute the GCD of coefficients of ``f`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x,y = ring("x,y", ZZ)
    >>> f = 2*x*y + 6*x + 4*y + 12

    >>> R.dmp_ground_content(f)
    2

    >>> R, x,y = ring("x,y", QQ)
    >>> f = 2*x*y + 6*x + 4*y + 12

    >>> R.dmp_ground_content(f)
    2

    """
    from sympy.polys.domains import QQ
    if not u:
        return dup_content(f, K)
    if dmp_zero_p(f, u):
        return K.zero
    cont, v = (K.zero, u - 1)
    if K == QQ:
        for c in f:
            cont = K.gcd(cont, dmp_ground_content(c, v, K))
    else:
        for c in f:
            cont = K.gcd(cont, dmp_ground_content(c, v, K))
            if K.is_one(cont):
                break
    return cont