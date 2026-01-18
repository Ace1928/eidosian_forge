from operator import mul
from functools import reduce
from sympy.core import oo
from sympy.core.symbol import Dummy
from sympy.polys import Poly, gcd, ZZ, cancel
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.integrals.risch import (gcdex_diophantine, frac_in, derivation,
def no_cancel_b_small(b, c, n, DE):
    """
    Poly Risch Differential Equation - No cancellation: deg(b) small enough.

    Explanation
    ===========

    Given a derivation D on k[t], ``n`` either an integer or +oo, and ``b``,``c``
    in k[t] with deg(b) < deg(D) - 1 and either D == d/dt or
    deg(D) >= 2, either raise NonElementaryIntegralException, in which case the
    equation Dq + b*q == c has no solution of degree at most n in k[t],
    or a solution q in k[t] of this equation with deg(q) <= n, or the
    tuple (h, b0, c0) such that h in k[t], b0, c0, in k, and for any
    solution q in k[t] of degree at most n of Dq + bq == c, y == q - h
    is a solution in k of Dy + b0*y == c0.
    """
    q = Poly(0, DE.t)
    while not c.is_zero:
        if n == 0:
            m = 0
        else:
            m = c.degree(DE.t) - DE.d.degree(DE.t) + 1
        if not 0 <= m <= n:
            raise NonElementaryIntegralException
        if m > 0:
            p = Poly(c.as_poly(DE.t).LC() / (m * DE.d.as_poly(DE.t).LC()) * DE.t ** m, DE.t, expand=False)
        else:
            if b.degree(DE.t) != c.degree(DE.t):
                raise NonElementaryIntegralException
            if b.degree(DE.t) == 0:
                return (q, b.as_poly(DE.T[DE.level - 1]), c.as_poly(DE.T[DE.level - 1]))
            p = Poly(c.as_poly(DE.t).LC() / b.as_poly(DE.t).LC(), DE.t, expand=False)
        q = q + p
        n = m - 1
        c = c - derivation(p, DE) - b * p
    return q