from sympy.core.singleton import S
from sympy.strategies.rl import (
from sympy.core.basic import Basic
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.symbol import symbols
from sympy.abc import x
def test_distribute_add_mul():
    x, y = symbols('x, y')
    expr = Mul(2, Add(x, y), evaluate=False)
    expected = Add(Mul(2, x), Mul(2, y))
    distribute_mul = distribute(Mul, Add)
    assert distribute_mul(expr) == expected