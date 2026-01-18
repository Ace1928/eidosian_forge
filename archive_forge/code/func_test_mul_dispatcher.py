from sympy.core.expr import Expr
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.operations import AssocOp, LatticeOp
from sympy.testing.pytest import raises
from sympy.core.sympify import SympifyError
from sympy.core.add import Add, add
from sympy.core.mul import Mul, mul
def test_mul_dispatcher():

    class NewBase(Expr):

        @property
        def _mul_handler(self):
            return NewMul

    class NewMul(NewBase, Mul):
        pass
    mul.register_handlerclass((Mul, NewMul), NewMul)
    a, b = (Symbol('a'), NewBase())
    assert mul(1, 2) == Mul(1, 2)
    assert mul(a, a) == Mul(a, a)
    assert mul(a, b, a) == NewMul(a ** 2, b)