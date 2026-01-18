from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.polyerrors import (
from sympy.utilities import variations
from math import ceil as _ceil, log as _log
def dup_trunc(f, p, K):
    """
    Reduce a ``K[x]`` polynomial modulo a constant ``p`` in ``K``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_trunc(2*x**3 + 3*x**2 + 5*x + 7, ZZ(3))
    -x**3 - x + 1

    """
    if K.is_ZZ:
        g = []
        for c in f:
            c = c % p
            if c > p // 2:
                g.append(c - p)
            else:
                g.append(c)
    else:
        g = [c % p for c in f]
    return dup_strip(g)