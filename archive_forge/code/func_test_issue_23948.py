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
def test_issue_23948():
    f = (((-2 * x ** 5 + 28 * x ** 4 - 144 * x ** 3 + 324 * x ** 2 - 270 * x) * log(x) ** 2 + (-4 * x ** 6 + 56 * x ** 5 - 288 * x ** 4 + 648 * x ** 3 - 540 * x ** 2) * log(x) + (2 * x ** 5 - 28 * x ** 4 + 144 * x ** 3 - 324 * x ** 2 + 270 * x) * exp(x) + (2 * x ** 5 - 28 * x ** 4 + 144 * x ** 3 - 324 * x ** 2 + 270 * x) * log(5) - 2 * x ** 7 + 26 * x ** 6 - 116 * x ** 5 + 180 * x ** 4 + 54 * x ** 3 - 270 * x ** 2) * log(-log(x) ** 2 - 2 * x * log(x) + exp(x) + log(5) - x ** 2 - x) ** 2 + ((4 * x ** 5 - 44 * x ** 4 + 168 * x ** 3 - 216 * x ** 2 - 108 * x + 324) * log(x) + (-2 * x ** 5 + 24 * x ** 4 - 108 * x ** 3 + 216 * x ** 2 - 162 * x) * exp(x) + 4 * x ** 6 - 42 * x ** 5 + 144 * x ** 4 - 108 * x ** 3 - 324 * x ** 2 + 486 * x) * log(-log(x) ** 2 - 2 * x * log(x) + exp(x) + log(5) - x ** 2 - x)) / (x * exp(x) ** 2 * log(x) ** 2 + 2 * x ** 2 * exp(x) ** 2 * log(x) - x * exp(x) ** 3 + (-x * log(5) + x ** 3 + x ** 2) * exp(x) ** 2)
    F = (x ** 4 - 12 * x ** 3 + 54 * x ** 2 - 108 * x + 81) * exp(-2 * x) * log(-x ** 2 - 2 * x * log(x) - x + exp(x) - log(x) ** 2 + log(5)) ** 2
    assert risch_integrate(f, x) == F