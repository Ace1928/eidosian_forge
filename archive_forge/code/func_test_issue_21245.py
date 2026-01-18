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
def test_issue_21245():
    fi = (1 + sqrt(5)) / 2
    assert (1 / (1 - x - x ** 2)).series(x, 1 / fi, 1).factor() == (-4812 - 2152 * sqrt(5) + 1686 * x + 754 * sqrt(5) * x + O((x - 2 / (1 + sqrt(5))) ** 2, (x, 2 / (1 + sqrt(5))))) / ((1 + sqrt(5)) * (20 + 9 * sqrt(5)) ** 2 * (x + sqrt(5) * x - 2))