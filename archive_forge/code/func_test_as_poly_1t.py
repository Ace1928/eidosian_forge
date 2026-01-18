from sympy.core.function import (Function, Lambda, diff, expand_log)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan, sin, tan)
from sympy.polys.polytools import (Poly, cancel, factor)
from sympy.integrals.risch import (gcdex_diophantine, frac_in, as_poly_1t,
from sympy.testing.pytest import raises
from sympy.abc import x, t, nu, z, a, y
def test_as_poly_1t():
    assert as_poly_1t(2 / t + t, t, z) in [Poly(t + 2 * z, t, z), Poly(t + 2 * z, z, t)]
    assert as_poly_1t(2 / t + 3 / t ** 2, t, z) in [Poly(2 * z + 3 * z ** 2, t, z), Poly(2 * z + 3 * z ** 2, z, t)]
    assert as_poly_1t(2 / ((exp(2) + 1) * t), t, z) in [Poly(2 / (exp(2) + 1) * z, t, z), Poly(2 / (exp(2) + 1) * z, z, t)]
    assert as_poly_1t(2 / ((exp(2) + 1) * t) + t, t, z) in [Poly(t + 2 / (exp(2) + 1) * z, t, z), Poly(t + 2 / (exp(2) + 1) * z, z, t)]
    assert as_poly_1t(S.Zero, t, z) == Poly(0, t, z)