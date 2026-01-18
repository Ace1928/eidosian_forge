from sympy.core.numbers import Rational
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.elementary.exponential import exp
def test_cmp_bug2():

    class T:
        pass
    t = T()
    assert not Symbol == t
    assert Symbol != t