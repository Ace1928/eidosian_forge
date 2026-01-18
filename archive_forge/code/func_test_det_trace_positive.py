from sympy.assumptions.ask import (Q, ask)
from sympy.core.symbol import Symbol
from sympy.matrices.expressions.diagonal import (DiagMatrix, DiagonalMatrix)
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions import (MatrixSymbol, Identity, ZeroMatrix,
from sympy.matrices.expressions.factorizations import LofLU
from sympy.testing.pytest import XFAIL
def test_det_trace_positive():
    X = MatrixSymbol('X', 4, 4)
    assert ask(Q.positive(Trace(X)), Q.positive_definite(X))
    assert ask(Q.positive(Determinant(X)), Q.positive_definite(X))