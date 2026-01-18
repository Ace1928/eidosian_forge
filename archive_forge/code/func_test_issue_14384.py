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
def test_issue_14384():
    x, a = symbols('x a')
    assert series(x ** a, x) == x ** a
    assert series(x ** (-2 * a), x) == x ** (-2 * a)
    assert series(exp(a * log(x)), x) == exp(a * log(x))
    assert series(x ** I, x) == x ** I
    assert series(x ** (I + 1), x) == x ** (1 + I)
    assert series(exp(I * log(x)), x) == exp(I * log(x))