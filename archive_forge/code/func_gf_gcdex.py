from math import ceil as _ceil, sqrt as _sqrt, prod
from sympy.core.random import uniform
from sympy.external.gmpy import SYMPY_INTS
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import _sort_factors
def gf_gcdex(f, g, p, K):
    """
    Extended Euclidean Algorithm in ``GF(p)[x]``.

    Given polynomials ``f`` and ``g`` in ``GF(p)[x]``, computes polynomials
    ``s``, ``t`` and ``h``, such that ``h = gcd(f, g)`` and ``s*f + t*g = h``.
    The typical application of EEA is solving polynomial diophantine equations.

    Consider polynomials ``f = (x + 7) (x + 1)``, ``g = (x + 7) (x**2 + 1)``
    in ``GF(11)[x]``. Application of Extended Euclidean Algorithm gives::

       >>> from sympy.polys.domains import ZZ
       >>> from sympy.polys.galoistools import gf_gcdex, gf_mul, gf_add

       >>> s, t, g = gf_gcdex(ZZ.map([1, 8, 7]), ZZ.map([1, 7, 1, 7]), 11, ZZ)
       >>> s, t, g
       ([5, 6], [6], [1, 7])

    As result we obtained polynomials ``s = 5*x + 6`` and ``t = 6``, and
    additionally ``gcd(f, g) = x + 7``. This is correct because::

       >>> S = gf_mul(s, ZZ.map([1, 8, 7]), 11, ZZ)
       >>> T = gf_mul(t, ZZ.map([1, 7, 1, 7]), 11, ZZ)

       >>> gf_add(S, T, 11, ZZ) == [1, 7]
       True

    References
    ==========

    .. [1] [Gathen99]_

    """
    if not (f or g):
        return ([K.one], [], [])
    p0, r0 = gf_monic(f, p, K)
    p1, r1 = gf_monic(g, p, K)
    if not f:
        return ([], [K.invert(p1, p)], r1)
    if not g:
        return ([K.invert(p0, p)], [], r0)
    s0, s1 = ([K.invert(p0, p)], [])
    t0, t1 = ([], [K.invert(p1, p)])
    while True:
        Q, R = gf_div(r0, r1, p, K)
        if not R:
            break
        (lc, r1), r0 = (gf_monic(R, p, K), r1)
        inv = K.invert(lc, p)
        s = gf_sub_mul(s0, s1, Q, p, K)
        t = gf_sub_mul(t0, t1, Q, p, K)
        s1, s0 = (gf_mul_ground(s, inv, p, K), s1)
        t1, t0 = (gf_mul_ground(t, inv, p, K), t1)
    return (s1, t1, r1)