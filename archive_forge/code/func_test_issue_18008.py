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
def test_issue_18008():
    y = x * (1 + x * (1 - x)) / (1 + x * (1 - x) - (1 - x) * (1 - x))
    assert y.series(x, oo, n=4) == -9 / (32 * x ** 3) - 3 / (16 * x ** 2) - 1 / (8 * x) + S(1) / 4 + x / 2 + O(x ** (-4), (x, oo))