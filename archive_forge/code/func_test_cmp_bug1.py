from sympy.core.numbers import Rational
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.elementary.exponential import exp
def test_cmp_bug1():

    class T:
        pass
    t = T()
    x = Symbol('x')
    assert not x == t
    assert x != t