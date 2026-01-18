from sympy.core.numbers import (E, Rational, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core import Add, Mul, Pow
from sympy.core.expr import unchanged
from sympy.testing.pytest import raises, XFAIL
from sympy.abc import x
def test_intersection_AccumBounds():
    assert B(0, 3).intersection(B(1, 2)) == B(1, 2)
    assert B(0, 3).intersection(B(1, 4)) == B(1, 3)
    assert B(0, 3).intersection(B(-1, 2)) == B(0, 2)
    assert B(0, 3).intersection(B(-1, 4)) == B(0, 3)
    assert B(0, 1).intersection(B(2, 3)) == S.EmptySet
    raises(TypeError, lambda: B(0, 3).intersection(1))