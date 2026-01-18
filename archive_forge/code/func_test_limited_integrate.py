from sympy.integrals.risch import DifferentialExtension, derivation
from sympy.integrals.prde import (prde_normal_denom, prde_special_denom,
from sympy.polys.polymatrix import PolyMatrix as Matrix
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.polytools import Poly
from sympy.abc import x, t, n
def test_limited_integrate():
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    G = [(Poly(x, x), Poly(x + 1, x))]
    assert limited_integrate(Poly(-(1 + x + 5 * x ** 2 - 3 * x ** 3), x), Poly(1 - x - x ** 2 + x ** 3, x), G, DE) == ((Poly(x ** 2 - x + 2, x), Poly(x - 1, x, domain='QQ')), [2])
    G = [(Poly(1, x), Poly(x, x))]
    assert limited_integrate(Poly(5 * x ** 2, x), Poly(3, x), G, DE) == ((Poly(5 * x ** 3 / 9, x), Poly(1, x, domain='QQ')), [0])