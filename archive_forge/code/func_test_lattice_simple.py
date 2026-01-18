from sympy.core.expr import Expr
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.operations import AssocOp, LatticeOp
from sympy.testing.pytest import raises
from sympy.core.sympify import SympifyError
from sympy.core.add import Add, add
from sympy.core.mul import Mul, mul
def test_lattice_simple():
    assert join(join(2, 3), 4) == join(2, join(3, 4))
    assert join(2, 3) == join(3, 2)
    assert join(0, 2) == 0
    assert join(1, 2) == 2
    assert join(2, 2) == 2
    assert join(join(2, 3), 4) == join(2, 3, 4)
    assert join() == 1
    assert join(4) == 4
    assert join(1, 4, 2, 3, 1, 3, 2) == join(2, 3, 4)