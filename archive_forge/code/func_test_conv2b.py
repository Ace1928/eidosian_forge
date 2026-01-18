from symengine import (Symbol, Integer, sympify, SympifyError, log,
from symengine.lib.symengine_wrapper import (Subs, Derivative, RealMPFR,
import unittest
@unittest.skipIf(not have_sympy, 'SymPy not installed')
def test_conv2b():
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    z = sympy.Symbol('z')
    e = x * y
    assert sympify(e) == Symbol('x') * Symbol('y')
    e = x * y * z
    assert sympify(e) == Symbol('x') * Symbol('y') * Symbol('z')