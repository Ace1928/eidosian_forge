from sympy.core.symbol import symbols
from sympy.matrices import (Matrix, MatrixSymbol, eye, Identity,
from sympy.matrices.expressions import MatrixExpr, MatAdd
from sympy.matrices.common import classof
from sympy.testing.pytest import raises
def test_matrix_symbol_MM():
    X = MatrixSymbol('X', 3, 3)
    Y = eye(3) + X
    assert Y[1, 1] == 1 + X[1, 1]