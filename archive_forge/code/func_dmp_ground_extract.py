from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.polyerrors import (
from sympy.utilities import variations
from math import ceil as _ceil, log as _log
def dmp_ground_extract(f, g, u, K):
    """
    Extract common content from a pair of polynomials in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_ground_extract(6*x*y + 12*x + 18, 4*x*y + 8*x + 12)
    (2, 3*x*y + 6*x + 9, 2*x*y + 4*x + 6)

    """
    fc = dmp_ground_content(f, u, K)
    gc = dmp_ground_content(g, u, K)
    gcd = K.gcd(fc, gc)
    if not K.is_one(gcd):
        f = dmp_quo_ground(f, gcd, u, K)
        g = dmp_quo_ground(g, gcd, u, K)
    return (gcd, f, g)