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
def test_DifferentialExtension_symlog():
    assert DifferentialExtension(log(x ** x), x)._important_attrs == (Poly(t0 * x, t1), Poly(1, t1), [Poly(1, x), Poly(1 / x, t0), Poly((t0 + 1) * t1, t1)], [x, t0, t1], [Lambda(i, log(i)), Lambda(i, exp(i * t0))], [(exp(x * log(x)), x ** x)], [None, 'log', 'exp'], [None, x, t0 * x])
    assert DifferentialExtension(log(x ** y), x)._important_attrs == (Poly(y * t0, t0), Poly(1, t0), [Poly(1, x), Poly(1 / x, t0)], [x, t0], [Lambda(i, log(i))], [(y * log(x), log(x ** y))], [None, 'log'], [None, x])
    assert DifferentialExtension(log(sqrt(x)), x)._important_attrs == (Poly(t0, t0), Poly(2, t0), [Poly(1, x), Poly(1 / x, t0)], [x, t0], [Lambda(i, log(i))], [(log(x) / 2, log(sqrt(x)))], [None, 'log'], [None, x])