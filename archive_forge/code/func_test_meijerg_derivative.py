from sympy.core.containers import Tuple
from sympy.core.function import Derivative
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import (appellf1, hyper, meijerg)
from sympy.series.order import O
from sympy.abc import x, z, k
from sympy.series.limits import limit
from sympy.testing.pytest import raises, slow
from sympy.core.random import (
def test_meijerg_derivative():
    assert meijerg([], [1, 1], [0, 0, x], [], z).diff(x) == log(z) * meijerg([], [1, 1], [0, 0, x], [], z) + 2 * meijerg([], [1, 1, 1], [0, 0, x, 0], [], z)
    y = randcplx()
    a = 5
    assert td(meijerg([x], [], [], [], y), x)
    assert td(meijerg([x ** 2], [], [], [], y), x)
    assert td(meijerg([], [x], [], [], y), x)
    assert td(meijerg([], [], [x], [], y), x)
    assert td(meijerg([], [], [], [x], y), x)
    assert td(meijerg([x], [a], [a + 1], [], y), x)
    assert td(meijerg([x], [a + 1], [a], [], y), x)
    assert td(meijerg([x, a], [], [], [a + 1], y), x)
    assert td(meijerg([x, a + 1], [], [], [a], y), x)
    b = Rational(3, 2)
    assert td(meijerg([a + 2], [b], [b - 3, x], [a], y), x)