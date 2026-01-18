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
def test_doit_matrix():
    X = ImmutableMatrix([[1, 2], [3, 4]])
    assert MatPow(X, 0).doit() == ImmutableMatrix(Identity(2))
    assert MatPow(X, 1).doit() == X
    assert MatPow(X, 2).doit() == X ** 2
    assert MatPow(X, -1).doit() == X.inv()
    assert MatPow(X, -2).doit() == X.inv() ** 2
    assert MatPow(ImmutableMatrix([4]), S.Half).doit() == ImmutableMatrix([2])
    X = ImmutableMatrix([[0, 2], [0, 4]])
    raises(ValueError, lambda: MatPow(X, -1).doit())
    raises(ValueError, lambda: MatPow(X, -2).doit())