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
def test_DifferentialExtension_log():
    assert DifferentialExtension(log(x) * log(x + 1) * log(2 * x ** 2 + 2 * x), x)._important_attrs == (Poly(t0 * t1 ** 2 + (t0 * log(2) + t0 ** 2) * t1, t1), Poly(1, t1), [Poly(1, x), Poly(1 / x, t0), Poly(1 / (x + 1), t1, expand=False)], [x, t0, t1], [Lambda(i, log(i)), Lambda(i, log(i + 1))], [], [None, 'log', 'log'], [None, x, x + 1])
    assert DifferentialExtension(x ** x * log(x), x)._important_attrs == (Poly(t0 * t1, t1), Poly(1, t1), [Poly(1, x), Poly(1 / x, t0), Poly((1 + t0) * t1, t1)], [x, t0, t1], [Lambda(i, log(i)), Lambda(i, exp(t0 * i))], [(exp(x * log(x)), x ** x)], [None, 'log', 'exp'], [None, x, t0 * x])