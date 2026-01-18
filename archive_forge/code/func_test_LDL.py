from symengine import symbols, init_printing
from symengine.lib.symengine_wrapper import (DenseMatrix, Symbol, Integer,
from symengine.test_utilities import raises
import unittest
def test_LDL():
    A = DenseMatrix(3, 3, [4, 12, -16, 12, 37, -43, -16, -43, 98])
    L, D = A.LDL()
    assert L == DenseMatrix(3, 3, [1, 0, 0, 3, 1, 0, -4, 5, 1])
    assert D == DenseMatrix(3, 3, [4, 0, 0, 0, 1, 0, 0, 0, 9])