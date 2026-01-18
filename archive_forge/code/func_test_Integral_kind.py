from sympy.core.add import Add
from sympy.core.kind import NumberKind, UndefinedKind
from sympy.core.mul import Mul
from sympy.core.numbers import pi, zoo, I, AlgebraicNumber
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.integrals.integrals import Integral
from sympy.core.function import Derivative
from sympy.matrices import (Matrix, SparseMatrix, ImmutableMatrix,
def test_Integral_kind():
    A = MatrixSymbol('A', 2, 2)
    assert Integral(comm_x, comm_x).kind is NumberKind
    assert Integral(A, comm_x).kind is MatrixKind(NumberKind)