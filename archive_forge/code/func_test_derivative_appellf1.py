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
def test_derivative_appellf1():
    from sympy.core.function import diff
    a, b1, b2, c, x, y, z = symbols('a b1 b2 c x y z')
    assert diff(appellf1(a, b1, b2, c, x, y), x) == a * b1 * appellf1(a + 1, b2, b1 + 1, c + 1, y, x) / c
    assert diff(appellf1(a, b1, b2, c, x, y), y) == a * b2 * appellf1(a + 1, b1, b2 + 1, c + 1, x, y) / c
    assert diff(appellf1(a, b1, b2, c, x, y), z) == 0
    assert diff(appellf1(a, b1, b2, c, x, y), a) == Derivative(appellf1(a, b1, b2, c, x, y), a)