from operator import mul
from functools import reduce
from sympy.core import oo
from sympy.core.symbol import Dummy
from sympy.polys import Poly, gcd, ZZ, cancel
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.integrals.risch import (gcdex_diophantine, frac_in, derivation,
def cancel_primitive(b, c, n, DE):
    """
    Poly Risch Differential Equation - Cancellation: Primitive case.

    Explanation
    ===========

    Given a derivation D on k[t], n either an integer or +oo, ``b`` in k, and
    ``c`` in k[t] with Dt in k and ``b != 0``, either raise
    NonElementaryIntegralException, in which case the equation Dq + b*q == c
    has no solution of degree at most n in k[t], or a solution q in k[t] of
    this equation with deg(q) <= n.
    """
    from .prde import is_log_deriv_k_t_radical_in_field
    with DecrementLevel(DE):
        ba, bd = frac_in(b, DE.t)
        A = is_log_deriv_k_t_radical_in_field(ba, bd, DE)
        if A is not None:
            n, z = A
            if n == 1:
                raise NotImplementedError('is_deriv_in_field() is required to  solve this problem.')
    if c.is_zero:
        return c
    if n < c.degree(DE.t):
        raise NonElementaryIntegralException
    q = Poly(0, DE.t)
    while not c.is_zero:
        m = c.degree(DE.t)
        if n < m:
            raise NonElementaryIntegralException
        with DecrementLevel(DE):
            a2a, a2d = frac_in(c.LC(), DE.t)
            sa, sd = rischDE(ba, bd, a2a, a2d, DE)
        stm = Poly(sa.as_expr() / sd.as_expr() * DE.t ** m, DE.t, expand=False)
        q += stm
        n = m - 1
        c -= b * stm + derivation(stm, DE)
    return q