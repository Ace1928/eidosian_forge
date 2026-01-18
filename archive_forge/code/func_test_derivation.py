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
def test_derivation():
    p = Poly(4 * x ** 4 * t ** 5 + (-4 * x ** 3 - 4 * x ** 4) * t ** 4 + (-3 * x ** 2 + 2 * x ** 3) * t ** 3 + (2 * x + 7 * x ** 2 + 2 * x ** 3) * t ** 2 + (1 - 4 * x - 4 * x ** 2) * t - 1 + 2 * x, t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(-t ** 2 - 3 / (2 * x) * t + 1 / (2 * x), t)]})
    assert derivation(p, DE) == Poly(-20 * x ** 4 * t ** 6 + (2 * x ** 3 + 16 * x ** 4) * t ** 5 + (21 * x ** 2 + 12 * x ** 3) * t ** 4 + (x * Rational(7, 2) - 25 * x ** 2 - 12 * x ** 3) * t ** 3 + (-5 - x * Rational(15, 2) + 7 * x ** 2) * t ** 2 - (3 - 8 * x - 10 * x ** 2 - 4 * x ** 3) / (2 * x) * t + (1 - 4 * x ** 2) / (2 * x), t)
    assert derivation(Poly(1, t), DE) == Poly(0, t)
    assert derivation(Poly(t, t), DE) == DE.d
    assert derivation(Poly(t ** 2 + 1 / x * t + (1 - 2 * x) / (4 * x ** 2), t), DE) == Poly(-2 * t ** 3 - 4 / x * t ** 2 - (5 - 2 * x) / (2 * x ** 2) * t - (1 - 2 * x) / (2 * x ** 3), t, domain='ZZ(x)')
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 / x, t1), Poly(t, t)]})
    assert derivation(Poly(x * t * t1, t), DE) == Poly(t * t1 + x * t * t1 + t, t)
    assert derivation(Poly(x * t * t1, t), DE, coefficientD=True) == Poly((1 + t1) * t, t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    assert derivation(Poly(x, x), DE) == Poly(1, x)
    assert derivation((x + 1) / (x - 1), DE, basic=True) == -2 / (1 - 2 * x + x ** 2)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    assert derivation((t + 1) / (t - 1), DE, basic=True) == -2 * t / (1 - 2 * t + t ** 2)
    assert derivation(t + 1, DE, basic=True) == t