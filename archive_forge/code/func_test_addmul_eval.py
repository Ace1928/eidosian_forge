from sympy.core.function import Function
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, tan)
from sympy.testing.pytest import XFAIL
def test_addmul_eval():
    a = Symbol('a')
    b = Symbol('b')
    c = Rational(1)
    p = Rational(5)
    assert c + a + b * c + a - p == 2 * a + b + -4
    assert a * 2 + p + a == a * 2 + 5 + a
    assert a * 2 + p + a == 3 * a + 5
    assert a * 2 + a == 3 * a