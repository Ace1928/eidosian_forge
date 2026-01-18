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
def test_heurisch_fractions():
    assert heurisch(1 / x, x) == log(x)
    assert heurisch(1 / (2 + x), x) == log(x + 2)
    assert heurisch(1 / (x + sin(y)), x) == log(x + sin(y))
    assert heurisch(5 * x ** 5 / (2 * x ** 6 - 5), x) in [5 * log(2 * x ** 6 - 5) / 12, 5 * log(-2 * x ** 6 + 5) / 12]
    assert heurisch(5 * x ** 5 / (2 * x ** 6 + 5), x) == 5 * log(2 * x ** 6 + 5) / 12
    assert heurisch(1 / x ** 2, x) == -1 / x
    assert heurisch(-1 / x ** 5, x) == 1 / (4 * x ** 4)