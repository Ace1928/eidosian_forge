from symengine import (Symbol, Integer, sympify, SympifyError, log,
from symengine.lib.symengine_wrapper import (Subs, Derivative, RealMPFR,
import unittest
@unittest.skipIf(not have_sympy, 'SymPy not installed')
def test_conv1():
    x = Symbol('x')
    assert x._sympy_() == sympy.Symbol('x')
    assert x._sympy_() != sympy.Symbol('y')
    x = Symbol('y')
    assert x._sympy_() != sympy.Symbol('x')
    assert x._sympy_() == sympy.Symbol('y')