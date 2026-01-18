from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import (
def dup_sqf_part(f, K):
    """
    Returns square-free part of a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_sqf_part(x**3 - 3*x - 2)
    x**2 - x - 2

    """
    if K.is_FiniteField:
        return dup_gf_sqf_part(f, K)
    if not f:
        return f
    if K.is_negative(dup_LC(f, K)):
        f = dup_neg(f, K)
    gcd = dup_gcd(f, dup_diff(f, 1, K), K)
    sqf = dup_quo(f, gcd, K)
    if K.is_Field:
        return dup_monic(sqf, K)
    else:
        return dup_primitive(sqf, K)[1]