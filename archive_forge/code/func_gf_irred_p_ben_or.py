from math import ceil as _ceil, sqrt as _sqrt, prod
from sympy.core.random import uniform
from sympy.external.gmpy import SYMPY_INTS
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import _sort_factors
def gf_irred_p_ben_or(f, p, K):
    """
    Ben-Or's polynomial irreducibility test over finite fields.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_irred_p_ben_or

    >>> gf_irred_p_ben_or(ZZ.map([1, 4, 2, 2, 3, 2, 4, 1, 4, 0, 4]), 5, ZZ)
    True
    >>> gf_irred_p_ben_or(ZZ.map([3, 2, 4]), 5, ZZ)
    False

    """
    n = gf_degree(f)
    if n <= 1:
        return True
    _, f = gf_monic(f, p, K)
    if n < 5:
        H = h = gf_pow_mod([K.one, K.zero], p, f, p, K)
        for i in range(0, n // 2):
            g = gf_sub(h, [K.one, K.zero], p, K)
            if gf_gcd(f, g, p, K) == [K.one]:
                h = gf_compose_mod(h, H, f, p, K)
            else:
                return False
    else:
        b = gf_frobenius_monomial_base(f, p, K)
        H = h = gf_frobenius_map([K.one, K.zero], f, b, p, K)
        for i in range(0, n // 2):
            g = gf_sub(h, [K.one, K.zero], p, K)
            if gf_gcd(f, g, p, K) == [K.one]:
                h = gf_frobenius_map(h, f, b, p, K)
            else:
                return False
    return True