from symengine import symbols, init_printing
from symengine.lib.symengine_wrapper import (DenseMatrix, Symbol, Integer,
from symengine.test_utilities import raises
import unittest
def test_add_matrix():
    A = DenseMatrix(2, 2, [1, 2, 3, 4])
    B = DenseMatrix(2, 2, [1, 0, 0, 1])
    assert A.add_matrix(B) == DenseMatrix(2, 2, [2, 2, 3, 5])
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    d = Symbol('d')
    A = DenseMatrix(2, 2, [a + b, a - b, a, b])
    B = DenseMatrix(2, 2, [a - b, a + b, -a, b])
    assert A.add_matrix(B) == DenseMatrix(2, 2, [2 * a, 2 * a, 0, 2 * b])
    assert A + B == DenseMatrix(2, 2, [2 * a, 2 * a, 0, 2 * b])
    C = DenseMatrix(1, 2, [a, b])
    raises(ShapeError, lambda: A + C)