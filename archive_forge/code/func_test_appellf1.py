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
def test_appellf1():
    a, b1, b2, c, x, y = symbols('a b1 b2 c x y')
    assert appellf1(a, b2, b1, c, y, x) == appellf1(a, b1, b2, c, x, y)
    assert appellf1(a, b1, b1, c, y, x) == appellf1(a, b1, b1, c, x, y)
    assert appellf1(a, b1, b2, c, S.Zero, S.Zero) is S.One
    f = appellf1(a, b1, b2, c, S.Zero, S.Zero, evaluate=False)
    assert f.func is appellf1
    assert f.doit() is S.One