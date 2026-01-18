from symengine import (Symbol, Integer, sympify, SympifyError, log,
from symengine.lib.symengine_wrapper import (Subs, Derivative, RealMPFR,
import unittest
@unittest.skipIf(not have_sympy, 'SymPy not installed')
def test_conv2():
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    e = x * y
    assert e._sympy_() == sympy.Symbol('x') * sympy.Symbol('y')
    e = x * y * z
    assert e._sympy_() == sympy.Symbol('x') * sympy.Symbol('y') * sympy.Symbol('z')