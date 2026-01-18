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
def test_heurisch_wrapper():
    f = 1 / (y + x)
    assert heurisch_wrapper(f, x) == log(x + y)
    f = 1 / (y - x)
    assert heurisch_wrapper(f, x) == -log(x - y)
    f = 1 / ((y - x) * (y + x))
    assert heurisch_wrapper(f, x) == Piecewise((-log(x - y) / (2 * y) + log(x + y) / (2 * y), Ne(y, 0)), (1 / x, True))
    f = sqrt(x ** 2 / ((y - x) * (y + x)))
    assert heurisch_wrapper(f, x) == x * sqrt(-x ** 2 / (x ** 2 - y ** 2)) - y ** 2 * sqrt(-x ** 2 / (x ** 2 - y ** 2)) / x