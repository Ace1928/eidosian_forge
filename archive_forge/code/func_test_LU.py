from sympy.matrices.expressions.factorizations import lu, LofCholesky, qr, svd
from sympy.assumptions.ask import (Q, ask)
from sympy.core.symbol import Symbol
from sympy.matrices.expressions.matexpr import MatrixSymbol
def test_LU():
    L, U = lu(X)
    assert L.shape == U.shape == X.shape
    assert ask(Q.lower_triangular(L))
    assert ask(Q.upper_triangular(U))