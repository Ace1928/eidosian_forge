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
def test_DifferentialExtension_handle_first():
    assert DifferentialExtension(exp(x) * log(x), x, handle_first='log')._important_attrs == (Poly(t0 * t1, t1), Poly(1, t1), [Poly(1, x), Poly(1 / x, t0), Poly(t1, t1)], [x, t0, t1], [Lambda(i, log(i)), Lambda(i, exp(i))], [], [None, 'log', 'exp'], [None, x, x])
    assert DifferentialExtension(exp(x) * log(x), x, handle_first='exp')._important_attrs == (Poly(t0 * t1, t1), Poly(1, t1), [Poly(1, x), Poly(t0, t0), Poly(1 / x, t1)], [x, t0, t1], [Lambda(i, exp(i)), Lambda(i, log(i))], [], [None, 'exp', 'log'], [None, x, x])
    assert DifferentialExtension(-x ** x * log(x) ** 2 + x ** x - x ** x / x, x, handle_first='exp')._important_attrs == DifferentialExtension(-x ** x * log(x) ** 2 + x ** x - x ** x / x, x, handle_first='log')._important_attrs == (Poly((-1 + x - x * t0 ** 2) * t1, t1), Poly(x, t1), [Poly(1, x), Poly(1 / x, t0), Poly((1 + t0) * t1, t1)], [x, t0, t1], [Lambda(i, log(i)), Lambda(i, exp(t0 * i))], [(exp(x * log(x)), x ** x)], [None, 'log', 'exp'], [None, x, t0 * x])