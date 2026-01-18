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
@XFAIL
def test_AccumBounds_powf():
    nn = Symbol('nn', nonnegative=True)
    assert B(1 + nn, 2 + nn) ** B(1, 2) == B(1 + nn, (2 + nn) ** 2)
    i = Symbol('i', integer=True, negative=True)
    assert B(1, 2) ** i == B(2 ** i, 1)