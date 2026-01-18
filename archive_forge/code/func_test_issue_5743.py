from sympy.core.expr import unchanged
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational as R, pi)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.series.order import O
from sympy.simplify.radsimp import expand_numer
from sympy.core.function import expand, expand_multinomial, expand_power_base
from sympy.testing.pytest import raises
from sympy.core.random import verify_numerically
from sympy.abc import x, y, z
def test_issue_5743():
    assert (x * sqrt(x + y) * (1 + sqrt(x + y))).expand() == x ** 2 + x * y + x * sqrt(x + y)
    assert (x * sqrt(x + y) * (1 + x * sqrt(x + y))).expand() == x ** 3 + x ** 2 * y + x * sqrt(x + y)