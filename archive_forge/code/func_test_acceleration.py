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
def test_acceleration():
    e = (1 + 1 / n) ** n
    assert round(richardson(e, n, 10, 20).evalf(), 10) == round(E.evalf(), 10)
    A = Sum(Integer(-1) ** (k + 1) / k, (k, 1, n))
    assert round(shanks(A, n, 25).evalf(), 4) == round(log(2).evalf(), 4)
    assert round(shanks(A, n, 25, 5).evalf(), 10) == round(log(2).evalf(), 10)