from symengine import symbols, init_printing
from symengine.lib.symengine_wrapper import (DenseMatrix, Symbol, Integer,
from symengine.test_utilities import raises
import unittest
def test_FFLU():
    A = DenseMatrix(4, 4, [1, 2, 3, 4, 2, 2, 3, 4, 3, 3, 3, 4, 9, 8, 7, 6])
    L, U = A.FFLU()
    assert L == DenseMatrix(4, 4, [1, 0, 0, 0, 2, -2, 0, -0, 3, -3, 3, 0, 9, -10, 10, -10])
    assert U == DenseMatrix(4, 4, [1, 2, 3, 4, 0, -2, -3, -4, 0, 0, 3, 4, 0, 0, 0, -10])