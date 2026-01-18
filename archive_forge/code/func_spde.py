from operator import mul
from functools import reduce
from sympy.core import oo
from sympy.core.symbol import Dummy
from sympy.polys import Poly, gcd, ZZ, cancel
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.integrals.risch import (gcdex_diophantine, frac_in, derivation,
def spde(a, b, c, n, DE):
    """
    Rothstein's Special Polynomial Differential Equation algorithm.

    Explanation
    ===========

    Given a derivation D on k[t], an integer n and ``a``,``b``,``c`` in k[t] with
    ``a != 0``, either raise NonElementaryIntegralException, in which case the
    equation a*Dq + b*q == c has no solution of degree at most ``n`` in
    k[t], or return the tuple (B, C, m, alpha, beta) such that B, C,
    alpha, beta in k[t], m in ZZ, and any solution q in k[t] of degree
    at most n of a*Dq + b*q == c must be of the form
    q == alpha*h + beta, where h in k[t], deg(h) <= m, and Dh + B*h == C.

    This constitutes step 4 of the outline given in the rde.py docstring.
    """
    zero = Poly(0, DE.t)
    alpha = Poly(1, DE.t)
    beta = Poly(0, DE.t)
    while True:
        if c.is_zero:
            return (zero, zero, 0, zero, beta)
        if (n < 0) is True:
            raise NonElementaryIntegralException
        g = a.gcd(b)
        if not c.rem(g).is_zero:
            raise NonElementaryIntegralException
        a, b, c = (a.quo(g), b.quo(g), c.quo(g))
        if a.degree(DE.t) == 0:
            b = b.to_field().quo(a)
            c = c.to_field().quo(a)
            return (b, c, n, alpha, beta)
        r, z = gcdex_diophantine(b, a, c)
        b += derivation(a, DE)
        c = z - derivation(r, DE)
        n -= a.degree(DE.t)
        beta += alpha * r
        alpha *= a