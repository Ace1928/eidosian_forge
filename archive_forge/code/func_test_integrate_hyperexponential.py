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
def test_integrate_hyperexponential():
    a = Poly((1 + 2 * t1 + t1 ** 2 + 2 * t1 ** 3) * t ** 2 + (1 + t1 ** 2) * t + 1 + t1 ** 2, t)
    d = Poly(1, t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 + t1 ** 2, t1), Poly(t * (1 + t1 ** 2), t)], 'Tfuncs': [tan, Lambda(i, exp(tan(i)))]})
    assert integrate_hyperexponential(a, d, DE) == (exp(2 * tan(x)) * tan(x) + exp(tan(x)), 1 + t1 ** 2, True)
    a = Poly((t1 ** 3 + (x + 1) * t1 ** 2 + t1 + x + 2) * t, t)
    assert integrate_hyperexponential(a, d, DE) == ((x + tan(x)) * exp(tan(x)), 0, True)
    a = Poly(t, t)
    d = Poly(1, t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(2 * x * t, t)], 'Tfuncs': [Lambda(i, exp(x ** 2))]})
    assert integrate_hyperexponential(a, d, DE) == (0, NonElementaryIntegral(exp(x ** 2), x), False)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)], 'Tfuncs': [exp]})
    assert integrate_hyperexponential(a, d, DE) == (exp(x), 0, True)
    a = Poly(25 * t ** 6 - 10 * t ** 5 + 7 * t ** 4 - 8 * t ** 3 + 13 * t ** 2 + 2 * t - 1, t)
    d = Poly(25 * t ** 6 + 35 * t ** 4 + 11 * t ** 2 + 1, t)
    assert integrate_hyperexponential(a, d, DE) == (-(11 - 10 * exp(x)) / (5 + 25 * exp(2 * x)) + log(1 + exp(2 * x)), -1, True)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t0, t0), Poly(t0 * t, t)], 'Tfuncs': [exp, Lambda(i, exp(exp(i)))]})
    assert integrate_hyperexponential(Poly(2 * t0 * t ** 2, t), Poly(1, t), DE) == (exp(2 * exp(x)), 0, True)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t0, t0), Poly(-t0 * t, t)], 'Tfuncs': [exp, Lambda(i, exp(-exp(i)))]})
    assert integrate_hyperexponential(Poly(-27 * exp(9) - 162 * t0 * exp(9) + 27 * x * t0 * exp(9), t), Poly((36 * exp(18) + x ** 2 * exp(18) - 12 * x * exp(18)) * t, t), DE) == (27 * exp(exp(x)) / (-6 * exp(9) + x * exp(9)), 0, True)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)], 'Tfuncs': [exp]})
    assert integrate_hyperexponential(Poly(x ** 2 / 2 * t, t), Poly(1, t), DE) == ((2 - 2 * x + x ** 2) * exp(x) / 2, 0, True)
    assert integrate_hyperexponential(Poly(1 + t, t), Poly(t, t), DE) == (-exp(-x), 1, True)
    assert integrate_hyperexponential(Poly(x, t), Poly(t + 1, t), DE) == (0, NonElementaryIntegral(x / (1 + exp(x)), x), False)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 / x, t0), Poly(2 * x * t1, t1)], 'Tfuncs': [log, Lambda(i, exp(i ** 2))]})
    elem, nonelem, b = integrate_hyperexponential(Poly((8 * x ** 7 - 12 * x ** 5 + 6 * x ** 3 - x) * t1 ** 4 + (8 * t0 * x ** 7 - 8 * t0 * x ** 6 - 4 * t0 * x ** 5 + 2 * t0 * x ** 3 + 2 * t0 * x ** 2 - t0 * x + 24 * x ** 8 - 36 * x ** 6 - 4 * x ** 5 + 22 * x ** 4 + 4 * x ** 3 - 7 * x ** 2 - x + 1) * t1 ** 3 + (8 * t0 * x ** 8 - 4 * t0 * x ** 6 - 16 * t0 * x ** 5 - 2 * t0 * x ** 4 + 12 * t0 * x ** 3 + t0 * x ** 2 - 2 * t0 * x + 24 * x ** 9 - 36 * x ** 7 - 8 * x ** 6 + 22 * x ** 5 + 12 * x ** 4 - 7 * x ** 3 - 6 * x ** 2 + x + 1) * t1 ** 2 + (8 * t0 * x ** 8 - 8 * t0 * x ** 6 - 16 * t0 * x ** 5 + 6 * t0 * x ** 4 + 10 * t0 * x ** 3 - 2 * t0 * x ** 2 - t0 * x + 8 * x ** 10 - 12 * x ** 8 - 4 * x ** 7 + 2 * x ** 6 + 12 * x ** 5 + 3 * x ** 4 - 9 * x ** 3 - x ** 2 + 2 * x) * t1 + 8 * t0 * x ** 7 - 12 * t0 * x ** 6 - 4 * t0 * x ** 5 + 8 * t0 * x ** 4 - t0 * x ** 2 - 4 * x ** 7 + 4 * x ** 6 + 4 * x ** 5 - 4 * x ** 4 - x ** 3 + x ** 2, t1), Poly((8 * x ** 7 - 12 * x ** 5 + 6 * x ** 3 - x) * t1 ** 4 + (24 * x ** 8 + 8 * x ** 7 - 36 * x ** 6 - 12 * x ** 5 + 18 * x ** 4 + 6 * x ** 3 - 3 * x ** 2 - x) * t1 ** 3 + (24 * x ** 9 + 24 * x ** 8 - 36 * x ** 7 - 36 * x ** 6 + 18 * x ** 5 + 18 * x ** 4 - 3 * x ** 3 - 3 * x ** 2) * t1 ** 2 + (8 * x ** 10 + 24 * x ** 9 - 12 * x ** 8 - 36 * x ** 7 + 6 * x ** 6 + 18 * x ** 5 - x ** 4 - 3 * x ** 3) * t1 + 8 * x ** 10 - 12 * x ** 8 + 6 * x ** 6 - x ** 4, t1), DE)
    assert factor(elem) == -((x - 1) * log(x) / ((x + exp(x ** 2)) * (2 * x ** 2 - 1)))
    assert (nonelem, b) == (NonElementaryIntegral(exp(x ** 2) / (exp(x ** 2) + 1), x), False)