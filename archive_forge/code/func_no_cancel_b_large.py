from operator import mul
from functools import reduce
from sympy.core import oo
from sympy.core.symbol import Dummy
from sympy.polys import Poly, gcd, ZZ, cancel
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.integrals.risch import (gcdex_diophantine, frac_in, derivation,
def no_cancel_b_large(b, c, n, DE):
    """
    Poly Risch Differential Equation - No cancellation: deg(b) large enough.

    Explanation
    ===========

    Given a derivation D on k[t], ``n`` either an integer or +oo, and ``b``,``c``
    in k[t] with ``b != 0`` and either D == d/dt or
    deg(b) > max(0, deg(D) - 1), either raise NonElementaryIntegralException, in
    which case the equation ``Dq + b*q == c`` has no solution of degree at
    most n in k[t], or a solution q in k[t] of this equation with
    ``deg(q) < n``.
    """
    q = Poly(0, DE.t)
    while not c.is_zero:
        m = c.degree(DE.t) - b.degree(DE.t)
        if not 0 <= m <= n:
            raise NonElementaryIntegralException
        p = Poly(c.as_poly(DE.t).LC() / b.as_poly(DE.t).LC() * DE.t ** m, DE.t, expand=False)
        q = q + p
        n = m - 1
        c = c - derivation(p, DE) - b * p
    return q