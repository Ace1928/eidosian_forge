from sympy.assumptions.ask import (Q, ask)
from sympy.core.symbol import Symbol
from sympy.matrices.expressions.diagonal import (DiagMatrix, DiagonalMatrix)
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions import (MatrixSymbol, Identity, ZeroMatrix,
from sympy.matrices.expressions.factorizations import LofLU
from sympy.testing.pytest import XFAIL
def test_matrix_element_sets_slices_blocks():
    X = MatrixSymbol('X', 4, 4)
    assert ask(Q.integer_elements(X[:, 3]), Q.integer_elements(X))
    assert ask(Q.integer_elements(BlockMatrix([[X], [X]])), Q.integer_elements(X))