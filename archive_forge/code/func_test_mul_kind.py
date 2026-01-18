from sympy.core.add import Add
from sympy.core.kind import NumberKind, UndefinedKind
from sympy.core.mul import Mul
from sympy.core.numbers import pi, zoo, I, AlgebraicNumber
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.integrals.integrals import Integral
from sympy.core.function import Derivative
from sympy.matrices import (Matrix, SparseMatrix, ImmutableMatrix,
def test_mul_kind():
    assert Mul(2, comm_x, evaluate=False).kind is NumberKind
    assert Mul(2, 3, evaluate=False).kind is NumberKind
    assert Mul(noncomm_x, 2, evaluate=False).kind is UndefinedKind
    assert Mul(2, noncomm_x, evaluate=False).kind is UndefinedKind