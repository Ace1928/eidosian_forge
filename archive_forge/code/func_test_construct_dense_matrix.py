from symengine import (Symbol, Integer, sympify, SympifyError, log,
from symengine.lib.symengine_wrapper import (Subs, Derivative, RealMPFR,
import unittest
@unittest.skipIf(not have_sympy, 'SymPy not installed')
def test_construct_dense_matrix():
    A = sympy.Matrix([[1, 2], [3, 5]])
    B = DenseMatrix(A)
    assert B.shape == (2, 2)
    assert list(B) == [1, 2, 3, 5]