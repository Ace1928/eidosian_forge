from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.polyerrors import (
from sympy.utilities import variations
from math import ceil as _ceil, log as _log
def dup_monic(f, K):
    """
    Divide all coefficients by ``LC(f)`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x = ring("x", ZZ)
    >>> R.dup_monic(3*x**2 + 6*x + 9)
    x**2 + 2*x + 3

    >>> R, x = ring("x", QQ)
    >>> R.dup_monic(3*x**2 + 4*x + 2)
    x**2 + 4/3*x + 2/3

    """
    if not f:
        return f
    lc = dup_LC(f, K)
    if K.is_one(lc):
        return f
    else:
        return dup_exquo_ground(f, lc, K)