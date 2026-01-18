from sympy.core.numbers import (Float, Rational)
from sympy.core.symbol import Symbol
def test_ibasic():

    def s(a, b):
        x = a
        x += b
        x = a
        x -= b
        x = a
        x *= b
        x = a
        x /= b
    assert dotest(s)