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
def test_AccumBounds_div():
    assert B(-1, 3) / B(3, 4) == B(Rational(-1, 3), 1)
    assert B(-2, 4) / B(-3, 4) == B(-oo, oo)
    assert B(-3, -2) / B(-4, 0) == B(S.Half, oo)
    assert B(-3, -2) / B(-2, 1) == B(-oo, oo)
    assert B(2, 3) / B(-2, 2) == B(-oo, oo)
    assert B(-3, -2) / B(0, 4) == B(-oo, Rational(-1, 2))
    assert B(2, 4) / B(-3, 0) == B(-oo, Rational(-2, 3))
    assert B(2, 4) / B(0, 3) == B(Rational(2, 3), oo)
    assert B(0, 1) / B(0, 1) == B(0, oo)
    assert B(-1, 0) / B(0, 1) == B(-oo, 0)
    assert B(-1, 2) / B(-2, 2) == B(-oo, oo)
    assert 1 / B(-1, 2) == B(-oo, oo)
    assert 1 / B(0, 2) == B(S.Half, oo)
    assert -1 / B(0, 2) == B(-oo, Rational(-1, 2))
    assert 1 / B(-oo, 0) == B(-oo, 0)
    assert 1 / B(-1, 0) == B(-oo, -1)
    assert -2 / B(-oo, 0) == B(0, oo)
    assert 1 / B(-oo, -1) == B(-1, 0)
    assert B(1, 2) / a == Mul(B(1, 2), 1 / a, evaluate=False)
    assert B(1, 2) / 0 == B(1, 2) * zoo
    assert B(1, oo) / oo == B(0, oo)
    assert B(1, oo) / -oo == B(-oo, 0)
    assert B(-oo, -1) / oo == B(-oo, 0)
    assert B(-oo, -1) / -oo == B(0, oo)
    assert B(-oo, oo) / oo == B(-oo, oo)
    assert B(-oo, oo) / -oo == B(-oo, oo)
    assert B(-1, oo) / oo == B(0, oo)
    assert B(-1, oo) / -oo == B(-oo, 0)
    assert B(-oo, 1) / oo == B(-oo, 0)
    assert B(-oo, 1) / -oo == B(0, oo)