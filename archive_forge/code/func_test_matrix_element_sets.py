from sympy.assumptions.ask import (Q, ask)
from sympy.core.symbol import Symbol
from sympy.matrices.expressions.diagonal import (DiagMatrix, DiagonalMatrix)
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions import (MatrixSymbol, Identity, ZeroMatrix,
from sympy.matrices.expressions.factorizations import LofLU
from sympy.testing.pytest import XFAIL
def test_matrix_element_sets():
    X = MatrixSymbol('X', 4, 4)
    assert ask(Q.real(X[1, 2]), Q.real_elements(X))
    assert ask(Q.integer(X[1, 2]), Q.integer_elements(X))
    assert ask(Q.complex(X[1, 2]), Q.complex_elements(X))
    assert ask(Q.integer_elements(Identity(3)))
    assert ask(Q.integer_elements(ZeroMatrix(3, 3)))
    assert ask(Q.integer_elements(OneMatrix(3, 3)))
    from sympy.matrices.expressions.fourier import DFT
    assert ask(Q.complex_elements(DFT(3)))