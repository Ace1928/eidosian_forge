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
def test_DifferentialExtension_exp():
    assert DifferentialExtension(exp(x) + exp(x ** 2), x)._important_attrs == (Poly(t1 + t0, t1), Poly(1, t1), [Poly(1, x), Poly(t0, t0), Poly(2 * x * t1, t1)], [x, t0, t1], [Lambda(i, exp(i)), Lambda(i, exp(i ** 2))], [], [None, 'exp', 'exp'], [None, x, x ** 2])
    assert DifferentialExtension(exp(x) + exp(2 * x), x)._important_attrs == (Poly(t0 ** 2 + t0, t0), Poly(1, t0), [Poly(1, x), Poly(t0, t0)], [x, t0], [Lambda(i, exp(i))], [], [None, 'exp'], [None, x])
    assert DifferentialExtension(exp(x) + exp(x / 2), x)._important_attrs == (Poly(t0 ** 2 + t0, t0), Poly(1, t0), [Poly(1, x), Poly(t0 / 2, t0)], [x, t0], [Lambda(i, exp(i / 2))], [], [None, 'exp'], [None, x / 2])
    assert DifferentialExtension(exp(x) + exp(x ** 2) + exp(x + x ** 2), x)._important_attrs == (Poly((1 + t0) * t1 + t0, t1), Poly(1, t1), [Poly(1, x), Poly(t0, t0), Poly(2 * x * t1, t1)], [x, t0, t1], [Lambda(i, exp(i)), Lambda(i, exp(i ** 2))], [], [None, 'exp', 'exp'], [None, x, x ** 2])
    assert DifferentialExtension(exp(x) + exp(x ** 2) + exp(x + x ** 2 + 1), x)._important_attrs == (Poly((1 + S.Exp1 * t0) * t1 + t0, t1), Poly(1, t1), [Poly(1, x), Poly(t0, t0), Poly(2 * x * t1, t1)], [x, t0, t1], [Lambda(i, exp(i)), Lambda(i, exp(i ** 2))], [], [None, 'exp', 'exp'], [None, x, x ** 2])
    assert DifferentialExtension(exp(x) + exp(x ** 2) + exp(x / 2 + x ** 2), x)._important_attrs == (Poly((t0 + 1) * t1 + t0 ** 2, t1), Poly(1, t1), [Poly(1, x), Poly(t0 / 2, t0), Poly(2 * x * t1, t1)], [x, t0, t1], [Lambda(i, exp(i / 2)), Lambda(i, exp(i ** 2))], [(exp(x / 2), sqrt(exp(x)))], [None, 'exp', 'exp'], [None, x / 2, x ** 2])
    assert DifferentialExtension(exp(x) + exp(x ** 2) + exp(x / 2 + x ** 2 + 3), x)._important_attrs == (Poly((t0 * exp(3) + 1) * t1 + t0 ** 2, t1), Poly(1, t1), [Poly(1, x), Poly(t0 / 2, t0), Poly(2 * x * t1, t1)], [x, t0, t1], [Lambda(i, exp(i / 2)), Lambda(i, exp(i ** 2))], [(exp(x / 2), sqrt(exp(x)))], [None, 'exp', 'exp'], [None, x / 2, x ** 2])
    assert DifferentialExtension(sqrt(exp(x)), x)._important_attrs == (Poly(t0, t0), Poly(1, t0), [Poly(1, x), Poly(t0 / 2, t0)], [x, t0], [Lambda(i, exp(i / 2))], [(exp(x / 2), sqrt(exp(x)))], [None, 'exp'], [None, x / 2])
    assert DifferentialExtension(exp(x / 2), x)._important_attrs == (Poly(t0, t0), Poly(1, t0), [Poly(1, x), Poly(t0 / 2, t0)], [x, t0], [Lambda(i, exp(i / 2))], [], [None, 'exp'], [None, x / 2])