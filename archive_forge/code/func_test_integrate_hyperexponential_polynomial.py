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
def test_integrate_hyperexponential_polynomial():
    p = Poly((-28 * x ** 11 * t0 - 6 * x ** 8 * t0 + 6 * x ** 9 * t0 - 15 * x ** 8 * t0 ** 2 + 15 * x ** 7 * t0 ** 2 + 84 * x ** 10 * t0 ** 2 - 140 * x ** 9 * t0 ** 3 - 20 * x ** 6 * t0 ** 3 + 20 * x ** 7 * t0 ** 3 - 15 * x ** 6 * t0 ** 4 + 15 * x ** 5 * t0 ** 4 + 140 * x ** 8 * t0 ** 4 - 84 * x ** 7 * t0 ** 5 - 6 * x ** 4 * t0 ** 5 + 6 * x ** 5 * t0 ** 5 + x ** 3 * t0 ** 6 - x ** 4 * t0 ** 6 + 28 * x ** 6 * t0 ** 6 - 4 * x ** 5 * t0 ** 7 + x ** 9 - x ** 10 + 4 * x ** 12) / (-8 * x ** 11 * t0 + 28 * x ** 10 * t0 ** 2 - 56 * x ** 9 * t0 ** 3 + 70 * x ** 8 * t0 ** 4 - 56 * x ** 7 * t0 ** 5 + 28 * x ** 6 * t0 ** 6 - 8 * x ** 5 * t0 ** 7 + x ** 4 * t0 ** 8 + x ** 12) * t1 ** 2 + (-28 * x ** 11 * t0 - 12 * x ** 8 * t0 + 12 * x ** 9 * t0 - 30 * x ** 8 * t0 ** 2 + 30 * x ** 7 * t0 ** 2 + 84 * x ** 10 * t0 ** 2 - 140 * x ** 9 * t0 ** 3 - 40 * x ** 6 * t0 ** 3 + 40 * x ** 7 * t0 ** 3 - 30 * x ** 6 * t0 ** 4 + 30 * x ** 5 * t0 ** 4 + 140 * x ** 8 * t0 ** 4 - 84 * x ** 7 * t0 ** 5 - 12 * x ** 4 * t0 ** 5 + 12 * x ** 5 * t0 ** 5 - 2 * x ** 4 * t0 ** 6 + 2 * x ** 3 * t0 ** 6 + 28 * x ** 6 * t0 ** 6 - 4 * x ** 5 * t0 ** 7 + 2 * x ** 9 - 2 * x ** 10 + 4 * x ** 12) / (-8 * x ** 11 * t0 + 28 * x ** 10 * t0 ** 2 - 56 * x ** 9 * t0 ** 3 + 70 * x ** 8 * t0 ** 4 - 56 * x ** 7 * t0 ** 5 + 28 * x ** 6 * t0 ** 6 - 8 * x ** 5 * t0 ** 7 + x ** 4 * t0 ** 8 + x ** 12) * t1 + (-2 * x ** 2 * t0 + 2 * x ** 3 * t0 + x * t0 ** 2 - x ** 2 * t0 ** 2 + x ** 3 - x ** 4) / (-4 * x ** 5 * t0 + 6 * x ** 4 * t0 ** 2 - 4 * x ** 3 * t0 ** 3 + x ** 2 * t0 ** 4 + x ** 6), t1, z, expand=False)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 / x, t0), Poly(2 * x * t1, t1)]})
    assert integrate_hyperexponential_polynomial(p, DE, z) == (Poly((x - t0) * t1 ** 2 + (-2 * t0 + 2 * x) * t1, t1), Poly(-2 * x * t0 + x ** 2 + t0 ** 2, t1), True)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t0, t0)]})
    assert integrate_hyperexponential_polynomial(Poly(0, t0), DE, z) == (Poly(0, t0), Poly(1, t0), True)