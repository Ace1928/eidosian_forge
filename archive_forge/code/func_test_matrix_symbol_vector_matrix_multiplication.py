from sympy.core.symbol import symbols
from sympy.matrices import (Matrix, MatrixSymbol, eye, Identity,
from sympy.matrices.expressions import MatrixExpr, MatAdd
from sympy.matrices.common import classof
from sympy.testing.pytest import raises
def test_matrix_symbol_vector_matrix_multiplication():
    A = MM * SV
    B = IM * SV
    assert A == B
    C = (SV.T * MM.T).T
    assert B == C
    D = (SV.T * IM.T).T
    assert C == D