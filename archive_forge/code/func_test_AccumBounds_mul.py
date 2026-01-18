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
def test_AccumBounds_mul():
    assert B(1, 2) * 2 == B(2, 4)
    assert 2 * B(1, 2) == B(2, 4)
    assert B(1, 2) * B(2, 3) == B(2, 6)
    assert B(0, 2) * B(2, oo) == B(0, oo)
    l, r = (B(-oo, oo), B(-a, a))
    assert l * r == B(-oo, oo)
    assert r * l == B(-oo, oo)
    l, r = (B(1, oo), B(-3, -2))
    assert l * r == B(-oo, -2)
    assert r * l == B(-oo, -2)
    assert B(1, 2) * 0 == 0
    assert B(1, oo) * 0 == B(0, oo)
    assert B(-oo, 1) * 0 == B(-oo, 0)
    assert B(-oo, oo) * 0 == B(-oo, oo)
    assert B(1, 2) * x == Mul(B(1, 2), x, evaluate=False)
    assert B(0, 2) * oo == B(0, oo)
    assert B(-2, 0) * oo == B(-oo, 0)
    assert B(0, 2) * -oo == B(-oo, 0)
    assert B(-2, 0) * -oo == B(0, oo)
    assert B(-1, 1) * oo == B(-oo, oo)
    assert B(-1, 1) * -oo == B(-oo, oo)
    assert B(-oo, oo) * oo == B(-oo, oo)