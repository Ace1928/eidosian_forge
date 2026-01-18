from sympy.functions.elementary.miscellaneous import sqrt
from sympy.simplify.powsimp import powsimp
from sympy.testing.pytest import raises
from sympy.core.expr import unchanged
from sympy.core import symbols, S
from sympy.matrices import Identity, MatrixSymbol, ImmutableMatrix, ZeroMatrix, OneMatrix, Matrix
from sympy.matrices.common import NonSquareMatrixError
from sympy.matrices.expressions import MatPow, MatAdd, MatMul
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.matexpr import MatrixElement
def test_as_explicit_matrix():
    A = ImmutableMatrix([[1, 2], [3, 4]])
    assert MatPow(A, 0).as_explicit() == ImmutableMatrix(Identity(2))
    assert MatPow(A, 1).as_explicit() == A
    assert MatPow(A, 2).as_explicit() == A ** 2
    assert MatPow(A, -1).as_explicit() == A.inv()
    assert MatPow(A, -2).as_explicit() == A.inv() ** 2
    A = ImmutableMatrix([4])
    assert MatPow(A, S.Half).as_explicit() == A ** S.Half