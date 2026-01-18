from sympy.core.symbol import symbols
from sympy.matrices import (Matrix, MatrixSymbol, eye, Identity,
from sympy.matrices.expressions import MatrixExpr, MatAdd
from sympy.matrices.common import classof
from sympy.testing.pytest import raises
def test_classof():
    A = Matrix(3, 3, range(9))
    B = ImmutableMatrix(3, 3, range(9))
    C = MatrixSymbol('C', 3, 3)
    assert classof(A, A) == Matrix
    assert classof(B, B) == ImmutableMatrix
    assert classof(A, B) == ImmutableMatrix
    assert classof(B, A) == ImmutableMatrix
    raises(TypeError, lambda: classof(A, C))