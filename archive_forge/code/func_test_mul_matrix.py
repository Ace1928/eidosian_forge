from symengine import symbols, init_printing
from symengine.lib.symengine_wrapper import (DenseMatrix, Symbol, Integer,
from symengine.test_utilities import raises
import unittest
def test_mul_matrix():
    A = DenseMatrix(2, 2, [1, 2, 3, 4])
    B = DenseMatrix(2, 2, [1, 0, 0, 1])
    assert A.mul_matrix(B) == A
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    d = Symbol('d')
    A = DenseMatrix(2, 2, [a, b, c, d])
    B = DenseMatrix(2, 2, [1, 0, 1, 0])
    assert A.mul_matrix(B) == DenseMatrix(2, 2, [a + b, 0, c + d, 0])
    assert A * B == DenseMatrix(2, 2, [a + b, 0, c + d, 0])
    assert A @ B == DenseMatrix(2, 2, [a + b, 0, c + d, 0])
    assert (A @ DenseMatrix(2, 1, [0] * 2)).shape == (2, 1)
    C = DenseMatrix(2, 3, [1, 2, 3, 2, 3, 4])
    D = DenseMatrix(3, 2, [3, 4, 4, 5, 5, 6])
    assert C.mul_matrix(D) == DenseMatrix(2, 2, [26, 32, 38, 47])
    raises(ShapeError, lambda: A * D)