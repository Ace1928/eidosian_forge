from symengine import symbols, init_printing
from symengine.lib.symengine_wrapper import (DenseMatrix, Symbol, Integer,
from symengine.test_utilities import raises
import unittest
def test_DenseMatrix_symbols():
    x, y, z = symbols('x y z')
    D = DenseMatrix(4, 4, [1, 0, 1, 0, 0, z, y, 0, z, 1, x, 1, 1, 1, 0, 0])
    assert D.get(1, 2) == y