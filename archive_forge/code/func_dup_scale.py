from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.polyerrors import (
from sympy.utilities import variations
from math import ceil as _ceil, log as _log
def dup_scale(f, a, K):
    """
    Evaluate efficiently composition ``f(a*x)`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_scale(x**2 - 2*x + 1, ZZ(2))
    4*x**2 - 4*x + 1

    """
    f, n, b = (list(f), len(f) - 1, a)
    for i in range(n - 1, -1, -1):
        f[i], b = (b * f[i], b * a)
    return f