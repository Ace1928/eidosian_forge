from symengine import (Symbol, Integer, sympify, SympifyError, log,
from symengine.lib.symengine_wrapper import (Subs, Derivative, RealMPFR,
import unittest
@unittest.skipIf(not have_sympy, 'SymPy not installed')
def test_conv10():
    A = DenseMatrix(1, 4, [Integer(1), Integer(2), Integer(3), Integer(4)])
    assert A._sympy_() == sympy.Matrix(1, 4, [sympy.Integer(1), sympy.Integer(2), sympy.Integer(3), sympy.Integer(4)])
    B = DenseMatrix(4, 1, [Symbol('x'), Symbol('y'), Symbol('z'), Symbol('t')])
    assert B._sympy_() == sympy.Matrix(4, 1, [sympy.Symbol('x'), sympy.Symbol('y'), sympy.Symbol('z'), sympy.Symbol('t')])
    C = DenseMatrix(2, 2, [Integer(5), Symbol('x'), function_symbol('f', Symbol('x')), 1 + I])
    assert C._sympy_() == sympy.Matrix([[5, sympy.Symbol('x')], [sympy.Function('f')(sympy.Symbol('x')), 1 + sympy.I]])