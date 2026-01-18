from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.polyerrors import (
from sympy.utilities import variations
from math import ceil as _ceil, log as _log
def dup_sign_variations(f, K):
    """
    Compute the number of sign variations of ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_sign_variations(x**4 - x**2 - x + 1)
    2

    """
    prev, k = (K.zero, 0)
    for coeff in f:
        if K.is_negative(coeff * prev):
            k += 1
        if coeff:
            prev = coeff
    return k