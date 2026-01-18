from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function)
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, asech)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin)
from sympy.functions.special.bessel import airyai
from sympy.functions.special.error_functions import erf
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import integrate
from sympy.series.formal import fps
from sympy.series.order import O
from sympy.series.formal import (rational_algorithm, FormalPowerSeries,
from sympy.testing.pytest import raises, XFAIL, slow
@slow
def test_fps__hyper():
    f = sin(x)
    assert fps(f, x).truncate() == x - x ** 3 / 6 + x ** 5 / 120 + O(x ** 6)
    f = cos(x)
    assert fps(f, x).truncate() == 1 - x ** 2 / 2 + x ** 4 / 24 + O(x ** 6)
    f = exp(x)
    assert fps(f, x).truncate() == 1 + x + x ** 2 / 2 + x ** 3 / 6 + x ** 4 / 24 + x ** 5 / 120 + O(x ** 6)
    f = atan(x)
    assert fps(f, x).truncate() == x - x ** 3 / 3 + x ** 5 / 5 + O(x ** 6)
    f = exp(acos(x))
    assert fps(f, x).truncate() == exp(pi / 2) - x * exp(pi / 2) + x ** 2 * exp(pi / 2) / 2 - x ** 3 * exp(pi / 2) / 3 + 5 * x ** 4 * exp(pi / 2) / 24 - x ** 5 * exp(pi / 2) / 6 + O(x ** 6)
    f = exp(acosh(x))
    assert fps(f, x).truncate() == I + x - I * x ** 2 / 2 - I * x ** 4 / 8 + O(x ** 6)
    f = atan(1 / x)
    assert fps(f, x).truncate() == pi / 2 - x + x ** 3 / 3 - x ** 5 / 5 + O(x ** 6)
    f = x * atan(x) - log(1 + x ** 2) / 2
    assert fps(f, x, rational=False).truncate() == x ** 2 / 2 - x ** 4 / 12 + O(x ** 6)
    f = log(1 + x)
    assert fps(f, x, rational=False).truncate() == x - x ** 2 / 2 + x ** 3 / 3 - x ** 4 / 4 + x ** 5 / 5 + O(x ** 6)
    f = airyai(x ** 2)
    assert fps(f, x).truncate() == 3 ** Rational(5, 6) * gamma(Rational(1, 3)) / (6 * pi) - 3 ** Rational(2, 3) * x ** 2 / (3 * gamma(Rational(1, 3))) + O(x ** 6)
    f = exp(x) * sin(x)
    assert fps(f, x).truncate() == x + x ** 2 + x ** 3 / 3 - x ** 5 / 30 + O(x ** 6)
    f = exp(x) * sin(x) / x
    assert fps(f, x).truncate() == 1 + x + x ** 2 / 3 - x ** 4 / 30 - x ** 5 / 90 + O(x ** 6)
    f = sin(x) * cos(x)
    assert fps(f, x).truncate() == x - 2 * x ** 3 / 3 + 2 * x ** 5 / 15 + O(x ** 6)