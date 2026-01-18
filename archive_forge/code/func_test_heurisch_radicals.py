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
def test_heurisch_radicals():
    assert heurisch(1 / sqrt(x), x) == 2 * sqrt(x)
    assert heurisch(1 / sqrt(x) ** 3, x) == -2 / sqrt(x)
    assert heurisch(sqrt(x) ** 3, x) == 2 * sqrt(x) ** 5 / 5
    assert heurisch(sin(x) * sqrt(cos(x)), x) == -2 * sqrt(cos(x)) ** 3 / 3
    y = Symbol('y')
    assert heurisch(sin(y * sqrt(x)), x) == 2 / y ** 2 * sin(y * sqrt(x)) - 2 * sqrt(x) * cos(y * sqrt(x)) / y
    assert heurisch_wrapper(sin(y * sqrt(x)), x) == Piecewise((-2 * sqrt(x) * cos(sqrt(x) * y) / y + 2 * sin(sqrt(x) * y) / y ** 2, Ne(y, 0)), (0, True))
    y = Symbol('y', positive=True)
    assert heurisch_wrapper(sin(y * sqrt(x)), x) == 2 / y ** 2 * sin(y * sqrt(x)) - 2 * sqrt(x) * cos(y * sqrt(x)) / y