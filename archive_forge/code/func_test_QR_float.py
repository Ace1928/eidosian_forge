from sympy.core.function import expand_mul
from sympy.core.numbers import I, Rational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.complexes import Abs
from sympy.simplify.simplify import simplify
from sympy.matrices.matrices import NonSquareMatrixError
from sympy.matrices import Matrix, zeros, eye, SparseMatrix
from sympy.abc import x, y, z
from sympy.testing.pytest import raises, slow
from sympy.testing.matrices import allclose
def test_QR_float():
    A = Matrix([[1, 1], [1, 1.01]])
    Q, R = A.QRdecomposition()
    assert allclose(Q * R, A)
    assert allclose(Q * Q.T, Matrix.eye(2))
    assert allclose(Q.T * Q, Matrix.eye(2))
    A = Matrix([[1, 1], [1, 1.001]])
    Q, R = A.QRdecomposition()
    assert allclose(Q * R, A)
    assert allclose(Q * Q.T, Matrix.eye(2))
    assert allclose(Q.T * Q, Matrix.eye(2))