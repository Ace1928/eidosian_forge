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
def test_meijerg_period():
    assert meijerg([], [1], [0], [], x).get_period() == 2 * pi
    assert meijerg([1], [], [], [0], x).get_period() == 2 * pi
    assert meijerg([], [], [0], [], x).get_period() == 2 * pi
    assert meijerg([], [], [0], [S.Half], x).get_period() == 2 * pi
    assert meijerg([], [], [S.Half], [0], x).get_period() == 4 * pi
    assert meijerg([1, 1], [], [1], [0], x).get_period() is oo