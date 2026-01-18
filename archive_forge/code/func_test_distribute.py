from sympy.core.singleton import S
from sympy.strategies.rl import (
from sympy.core.basic import Basic
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.symbol import symbols
from sympy.abc import x
def test_distribute():

    class T1(Basic):
        pass

    class T2(Basic):
        pass
    distribute_t12 = distribute(T1, T2)
    assert distribute_t12(T1(S(1), S(2), T2(S(3), S(4)), S(5))) == T2(T1(S(1), S(2), S(3), S(5)), T1(S(1), S(2), S(4), S(5)))
    assert distribute_t12(T1(S(1), S(2), S(3))) == T1(S(1), S(2), S(3))