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
def test_issue_7259():
    assert series(LambertW(x), x) == x - x ** 2 + 3 * x ** 3 / 2 - 8 * x ** 4 / 3 + 125 * x ** 5 / 24 + O(x ** 6)
    assert series(LambertW(x ** 2), x, n=8) == x ** 2 - x ** 4 + 3 * x ** 6 / 2 + O(x ** 8)
    assert series(LambertW(sin(x)), x, n=4) == x - x ** 2 + 4 * x ** 3 / 3 + O(x ** 4)