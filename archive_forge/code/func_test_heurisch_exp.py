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
def test_heurisch_exp():
    assert heurisch(exp(x), x) == exp(x)
    assert heurisch(exp(-x), x) == -exp(-x)
    assert heurisch(exp(17 * x), x) == exp(17 * x) / 17
    assert heurisch(x * exp(x), x) == x * exp(x) - exp(x)
    assert heurisch(x * exp(x ** 2), x) == exp(x ** 2) / 2
    assert heurisch(exp(-x ** 2), x) is None
    assert heurisch(2 ** x, x) == 2 ** x / log(2)
    assert heurisch(x * 2 ** x, x) == x * 2 ** x / log(2) - 2 ** x * log(2) ** (-2)
    assert heurisch(Integral(x ** z * y, (y, 1, 2), (z, 2, 3)).function, x) == x * x ** z * y / (z + 1)
    assert heurisch(Sum(x ** z, (z, 1, 2)).function, z) == x ** z / log(x)
    anti = -exp(z) / (sqrt(x - y) * exp(z * sqrt(x - y)) - exp(z * sqrt(x - y)))
    assert heurisch(exp(z) * exp(-z * sqrt(x - y)), z) == anti