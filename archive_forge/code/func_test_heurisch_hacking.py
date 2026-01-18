from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Ne
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (asinh, cosh, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin, tan)
from sympy.functions.special.bessel import (besselj, besselk, bessely, jn)
from sympy.functions.special.error_functions import erf
from sympy.integrals.integrals import Integral
from sympy.simplify.ratsimp import ratsimp
from sympy.simplify.simplify import simplify
from sympy.integrals.heurisch import components, heurisch, heurisch_wrapper
from sympy.testing.pytest import XFAIL, skip, slow, ON_CI
from sympy.integrals.integrals import integrate
def test_heurisch_hacking():
    assert heurisch(sqrt(1 + 7 * x ** 2), x, hints=[]) == x * sqrt(1 + 7 * x ** 2) / 2 + sqrt(7) * asinh(sqrt(7) * x) / 14
    assert heurisch(sqrt(1 - 7 * x ** 2), x, hints=[]) == x * sqrt(1 - 7 * x ** 2) / 2 + sqrt(7) * asin(sqrt(7) * x) / 14
    assert heurisch(1 / sqrt(1 + 7 * x ** 2), x, hints=[]) == sqrt(7) * asinh(sqrt(7) * x) / 7
    assert heurisch(1 / sqrt(1 - 7 * x ** 2), x, hints=[]) == sqrt(7) * asin(sqrt(7) * x) / 7
    assert heurisch(exp(-7 * x ** 2), x, hints=[]) == sqrt(7 * pi) * erf(sqrt(7) * x) / 14
    assert heurisch(1 / sqrt(9 - 4 * x ** 2), x, hints=[]) == asin(x * Rational(2, 3)) / 2
    assert heurisch(1 / sqrt(9 + 4 * x ** 2), x, hints=[]) == asinh(x * Rational(2, 3)) / 2
    assert heurisch(1 / sqrt(3 * x ** 2 - 4), x, hints=[]) == sqrt(3) * log(3 * x + sqrt(3) * sqrt(3 * x ** 2 - 4)) / 3