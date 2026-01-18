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
def test_fps__Add_expr():
    f = x * atan(x) - log(1 + x ** 2) / 2
    assert fps(f, x).truncate() == x ** 2 / 2 - x ** 4 / 12 + O(x ** 6)
    f = sin(x) + cos(x) - exp(x) + log(1 + x)
    assert fps(f, x).truncate() == x - 3 * x ** 2 / 2 - x ** 4 / 4 + x ** 5 / 5 + O(x ** 6)
    f = 1 / x + sin(x)
    assert fps(f, x).truncate() == 1 / x + x - x ** 3 / 6 + x ** 5 / 120 + O(x ** 6)
    f = sin(x) - cos(x) + 1 / (x - 1)
    assert fps(f, x).truncate() == -2 - x ** 2 / 2 - 7 * x ** 3 / 6 - 25 * x ** 4 / 24 - 119 * x ** 5 / 120 + O(x ** 6)