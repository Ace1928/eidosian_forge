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
def test_issue_18795():
    r = Symbol('r', real=True)
    a = B(-1, 1)
    c = B(7, oo)
    b = B(-oo, oo)
    assert c - tan(r) == B(7 - tan(r), oo)
    assert b + tan(r) == B(-oo, oo)
    assert (a + r) / a == B(-oo, oo) * B(r - 1, r + 1)
    assert (b + a) / a == B(-oo, oo)