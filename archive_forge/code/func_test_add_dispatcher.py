from sympy.core.expr import Expr
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.operations import AssocOp, LatticeOp
from sympy.testing.pytest import raises
from sympy.core.sympify import SympifyError
from sympy.core.add import Add, add
from sympy.core.mul import Mul, mul
def test_add_dispatcher():

    class NewBase(Expr):

        @property
        def _add_handler(self):
            return NewAdd

    class NewAdd(NewBase, Add):
        pass
    add.register_handlerclass((Add, NewAdd), NewAdd)
    a, b = (Symbol('a'), NewBase())
    assert add(1, 2) == Add(1, 2)
    assert add(a, a) == Add(a, a)
    assert add(a, b, a) == NewAdd(2 * a, b)