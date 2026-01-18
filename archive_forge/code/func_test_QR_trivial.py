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
def test_QR_trivial():
    A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q * R
    A = Matrix([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q * R
    A = Matrix([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]).T
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q * R
    A = Matrix([[0, 0, 0]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q * R
    A = Matrix([[0, 0, 0]]).T
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q * R
    A = Matrix([[0, 0, 0], [0, 0, 0]])
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q * R
    A = Matrix([[0, 0, 0], [0, 0, 0]]).T
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q * R
    A = Matrix([[0, 0, 0], [1, 2, 3]]).T
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q * R
    A = Matrix([[0, 0, 0, 0], [1, 2, 3, 4], [0, 0, 0, 0]]).T
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q * R
    A = Matrix([[0, 0, 0, 0], [1, 2, 3, 4], [0, 0, 0, 0], [2, 4, 6, 8]]).T
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q * R
    A = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 2, 3]]).T
    Q, R = A.QRdecomposition()
    assert Q.T * Q == eye(Q.cols)
    assert R.is_upper
    assert A == Q * R