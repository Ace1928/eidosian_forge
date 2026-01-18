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
def test_eval_nseries():
    a1, b1, a2, b2 = symbols('a1 b1 a2 b2')
    assert hyper((1, 2), (1, 2, 3), x ** 2)._eval_nseries(x, 7, None) == 1 + x ** 2 / 3 + x ** 4 / 24 + x ** 6 / 360 + O(x ** 7)
    assert exp(x)._eval_nseries(x, 7, None) == hyper((a1, b1), (a1, b1), x)._eval_nseries(x, 7, None)
    assert hyper((a1, a2), (b1, b2), x)._eval_nseries(z, 7, None) == hyper((a1, a2), (b1, b2), x) + O(z ** 7)