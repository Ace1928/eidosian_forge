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
def test_expand_radicals():
    a = (x + y) ** R(1, 2)
    assert (a ** 1).expand() == a
    assert (a ** 3).expand() == x * a + y * a
    assert (a ** 5).expand() == x ** 2 * a + 2 * x * y * a + y ** 2 * a
    assert (1 / a ** 1).expand() == 1 / a
    assert (1 / a ** 3).expand() == 1 / (x * a + y * a)
    assert (1 / a ** 5).expand() == 1 / (x ** 2 * a + 2 * x * y * a + y ** 2 * a)
    a = (x + y) ** R(1, 3)
    assert (a ** 1).expand() == a
    assert (a ** 2).expand() == a ** 2
    assert (a ** 4).expand() == x * a + y * a
    assert (a ** 5).expand() == x * a ** 2 + y * a ** 2
    assert (a ** 7).expand() == x ** 2 * a + 2 * x * y * a + y ** 2 * a