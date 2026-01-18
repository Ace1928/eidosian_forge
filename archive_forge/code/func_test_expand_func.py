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
def test_expand_func():
    from sympy.abc import a, b, c
    from sympy.core.function import expand_func
    a1, b1, c1 = (randcplx(), randcplx(), randcplx() + 5)
    assert expand_func(hyper([a, b], [c], 1)) == gamma(c) * gamma(-a - b + c) / (gamma(-a + c) * gamma(-b + c))
    assert abs(expand_func(hyper([a1, b1], [c1], 1)).n() - hyper([a1, b1], [c1], 1).n()) < 1e-10
    assert expand_func(hyper([], [], z)) == exp(z)
    assert expand_func(hyper([1, 2, 3], [], z)) == hyper([1, 2, 3], [], z)
    assert expand_func(meijerg([[1, 1], []], [[1], [0]], z)) == log(z + 1)
    assert expand_func(meijerg([[1, 1], []], [[], []], z)) == meijerg([[1, 1], []], [[], []], z)