from symengine import symbols, init_printing
from symengine.lib.symengine_wrapper import (DenseMatrix, Symbol, Integer,
from symengine.test_utilities import raises
import unittest
def test_FFLDU():
    A = DenseMatrix(3, 3, [1, 2, 3, 5, -3, 2, 6, 2, 1])
    L, D, U = A.FFLDU()
    assert L == DenseMatrix(3, 3, [1, 0, 0, 5, -13, 0, 6, -10, 1])
    assert D == DenseMatrix(3, 3, [1, 0, 0, 0, -13, 0, 0, 0, -13])
    assert U == DenseMatrix(3, 3, [1, 2, 3, 0, -13, -13, 0, 0, 91])