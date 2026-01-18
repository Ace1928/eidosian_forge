from sympy.assumptions.ask import (Q, ask)
from sympy.core.symbol import Symbol
from sympy.matrices.expressions.diagonal import (DiagMatrix, DiagonalMatrix)
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions import (MatrixSymbol, Identity, ZeroMatrix,
from sympy.matrices.expressions.factorizations import LofLU
from sympy.testing.pytest import XFAIL
def test_invertible_BlockMatrix():
    assert ask(Q.invertible(BlockMatrix([Identity(3)]))) == True
    assert ask(Q.invertible(BlockMatrix([ZeroMatrix(3, 3)]))) == False
    X = Matrix([[1, 2, 3], [3, 5, 4]])
    Y = Matrix([[4, 2, 7], [2, 3, 5]])
    assert ask(Q.invertible(BlockMatrix([[Matrix.ones(3, 3), Y.T], [X, Matrix.eye(2)]]))) == True
    assert ask(Q.invertible(BlockMatrix([[Y.T, Matrix.ones(3, 3)], [Matrix.eye(2), X]]))) == True
    assert ask(Q.invertible(BlockMatrix([[X, Matrix.eye(2)], [Matrix.ones(3, 3), Y.T]]))) == True
    assert ask(Q.invertible(BlockMatrix([[Matrix.eye(2), X], [Y.T, Matrix.ones(3, 3)]]))) == True