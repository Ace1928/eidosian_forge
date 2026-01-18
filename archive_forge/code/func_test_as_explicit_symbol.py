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
def test_as_explicit_symbol():
    X = MatrixSymbol('X', 2, 2)
    assert MatPow(X, 0).as_explicit() == ImmutableMatrix(Identity(2))
    assert MatPow(X, 1).as_explicit() == X.as_explicit()
    assert MatPow(X, 2).as_explicit() == X.as_explicit() ** 2
    assert MatPow(X, n).as_explicit() == ImmutableMatrix([[(X ** n)[0, 0], (X ** n)[0, 1]], [(X ** n)[1, 0], (X ** n)[1, 1]]])
    a = MatrixSymbol('a', 3, 1)
    b = MatrixSymbol('b', 3, 1)
    c = MatrixSymbol('c', 3, 1)
    expr = (a.T * b) ** S.Half
    assert expr.as_explicit() == Matrix([[sqrt(a[0, 0] * b[0, 0] + a[1, 0] * b[1, 0] + a[2, 0] * b[2, 0])]])
    expr = c * (a.T * b) ** S.Half
    m = sqrt(a[0, 0] * b[0, 0] + a[1, 0] * b[1, 0] + a[2, 0] * b[2, 0])
    assert expr.as_explicit() == Matrix([[c[0, 0] * m], [c[1, 0] * m], [c[2, 0] * m]])
    expr = (a * b.T) ** S.Half
    denom = sqrt(a[0, 0] * b[0, 0] + a[1, 0] * b[1, 0] + a[2, 0] * b[2, 0])
    expected = (a * b.T).as_explicit() / denom
    assert expr.as_explicit() == expected
    expr = X ** (-1)
    det = X[0, 0] * X[1, 1] - X[1, 0] * X[0, 1]
    expected = Matrix([[X[1, 1], -X[0, 1]], [-X[1, 0], X[0, 0]]]) / det
    assert expr.as_explicit() == expected
    expr = X ** m
    assert expr.as_explicit() == X.as_explicit() ** m