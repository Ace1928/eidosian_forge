from symengine import (Symbol, Integer, sympify, SympifyError, log,
from symengine.lib.symengine_wrapper import (Subs, Derivative, RealMPFR,
import unittest
@unittest.skipIf(not have_sympy, 'SymPy not installed')
def test_conv5():
    x = Integer(5)
    y = Integer(6)
    assert x._sympy_() == sympy.Integer(5)
    assert (x / y)._sympy_() == sympy.Integer(5) / sympy.Integer(6)