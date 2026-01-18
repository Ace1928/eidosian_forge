from symengine import (Symbol, Integer, sympify, SympifyError, log,
from symengine.lib.symengine_wrapper import (Subs, Derivative, RealMPFR,
import unittest
@unittest.skipIf(not have_sympy, 'SymPy not installed')
def test_conv1b():
    x = sympy.Symbol('x')
    assert sympify(x) == Symbol('x')
    assert sympify(x) != Symbol('y')
    x = sympy.Symbol('y')
    assert sympify(x) != Symbol('x')
    assert sympify(x) == Symbol('y')