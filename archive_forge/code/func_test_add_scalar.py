from symengine import symbols, init_printing
from symengine.lib.symengine_wrapper import (DenseMatrix, Symbol, Integer,
from symengine.test_utilities import raises
import unittest
def test_add_scalar():
    A = DenseMatrix(2, 2, [1, 2, 3, 4])
    a = Symbol('a')
    assert A.add_scalar(a) == DenseMatrix(2, 2, [1 + a, 2 + a, 3 + a, 4 + a])
    i5 = Integer(5)
    assert A.add_scalar(i5) == DenseMatrix(2, 2, [6, 7, 8, 9])
    raises(TypeError, lambda: A + 5)
    raises(TypeError, lambda: 5 + A)