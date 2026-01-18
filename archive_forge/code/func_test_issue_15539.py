from sympy.core.evalf import N
from sympy.core.function import (Derivative, Function, PoleError, Subs)
from sympy.core.numbers import (E, Float, Rational, oo, pi, I)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (atan, cos, sin)
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral, integrate
from sympy.series.order import O
from sympy.series.series import series
from sympy.abc import x, y, n, k
from sympy.testing.pytest import raises
from sympy.series.acceleration import richardson, shanks
from sympy.concrete.summations import Sum
from sympy.core.numbers import Integer
def test_issue_15539():
    assert series(atan(x), x, -oo) == -1 / (5 * x ** 5) + 1 / (3 * x ** 3) - 1 / x - pi / 2 + O(x ** (-6), (x, -oo))
    assert series(atan(x), x, oo) == -1 / (5 * x ** 5) + 1 / (3 * x ** 3) - 1 / x + pi / 2 + O(x ** (-6), (x, oo))