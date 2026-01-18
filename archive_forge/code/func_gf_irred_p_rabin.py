from math import ceil as _ceil, sqrt as _sqrt, prod
from sympy.core.random import uniform
from sympy.external.gmpy import SYMPY_INTS
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import _sort_factors
def gf_irred_p_rabin(f, p, K):
    """
    Rabin's polynomial irreducibility test over finite fields.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_irred_p_rabin

    >>> gf_irred_p_rabin(ZZ.map([1, 4, 2, 2, 3, 2, 4, 1, 4, 0, 4]), 5, ZZ)
    True
    >>> gf_irred_p_rabin(ZZ.map([3, 2, 4]), 5, ZZ)
    False

    """
    n = gf_degree(f)
    if n <= 1:
        return True
    _, f = gf_monic(f, p, K)
    x = [K.one, K.zero]
    from sympy.ntheory import factorint
    indices = {n // d for d in factorint(n)}
    b = gf_frobenius_monomial_base(f, p, K)
    h = b[1]
    for i in range(1, n):
        if i in indices:
            g = gf_sub(h, x, p, K)
            if gf_gcd(f, g, p, K) != [K.one]:
                return False
        h = gf_frobenius_map(h, f, b, p, K)
    return h == x