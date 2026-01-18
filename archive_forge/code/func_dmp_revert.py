from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.polyerrors import (
from sympy.utilities import variations
from math import ceil as _ceil, log as _log
def dmp_revert(f, g, u, K):
    """
    Compute ``f**(-1)`` mod ``x**n`` using Newton iteration.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x,y = ring("x,y", QQ)

    """
    if not u:
        return dup_revert(f, g, K)
    else:
        raise MultivariatePolynomialError(f, g)