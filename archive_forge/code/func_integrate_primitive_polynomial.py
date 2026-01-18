from the names used in Bronstein's book.
from types import GeneratorType
from functools import reduce
from sympy.core.function import Lambda
from sympy.core.mul import Mul
from sympy.core.numbers import ilcm, I, oo
from sympy.core.power import Pow
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.exponential import log, exp
from sympy.functions.elementary.hyperbolic import (cosh, coth, sinh,
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan, sin, cos,
from .integrals import integrate, Integral
from .heurisch import _symbols
from sympy.polys.polyerrors import DomainError, PolynomialError
from sympy.polys.polytools import (real_roots, cancel, Poly, gcd,
from sympy.polys.rootoftools import RootSum
from sympy.utilities.iterables import numbered_symbols
def integrate_primitive_polynomial(p, DE):
    """
    Integration of primitive polynomials.

    Explanation
    ===========

    Given a primitive monomial t over k, and ``p`` in k[t], return q in k[t],
    r in k, and a bool b in {True, False} such that r = p - Dq is in k if b is
    True, or r = p - Dq does not have an elementary integral over k(t) if b is
    False.
    """
    Zero = Poly(0, DE.t)
    q = Poly(0, DE.t)
    if not p.expr.has(DE.t):
        return (Zero, p, True)
    from .prde import limited_integrate
    while True:
        if not p.expr.has(DE.t):
            return (q, p, True)
        Dta, Dtb = frac_in(DE.d, DE.T[DE.level - 1])
        with DecrementLevel(DE):
            a = p.LC()
            aa, ad = frac_in(a, DE.t)
            try:
                rv = limited_integrate(aa, ad, [(Dta, Dtb)], DE)
                if rv is None:
                    raise NonElementaryIntegralException
                (ba, bd), c = rv
            except NonElementaryIntegralException:
                return (q, p, False)
        m = p.degree(DE.t)
        q0 = c[0].as_poly(DE.t) * Poly(DE.t ** (m + 1) / (m + 1), DE.t) + (ba.as_expr() / bd.as_expr()).as_poly(DE.t) * Poly(DE.t ** m, DE.t)
        p = p - derivation(q0, DE)
        q = q + q0