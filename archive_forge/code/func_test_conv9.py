from symengine import (Symbol, Integer, sympify, SympifyError, log,
from symengine.lib.symengine_wrapper import (Subs, Derivative, RealMPFR,
import unittest
@unittest.skipIf(not have_sympy, 'SymPy not installed')
def test_conv9():
    x = Symbol('x')
    y = Symbol('y')
    assert I._sympy_() == sympy.I
    assert (2 * I + 3)._sympy_() == 2 * sympy.I + 3
    assert (2 * I / 5 + Integer(3) / 5)._sympy_() == 2 * sympy.I / 5 + sympy.S(3) / 5
    assert (x * I + 3)._sympy_() == sympy.Symbol('x') * sympy.I + 3
    assert (x + I * y)._sympy_() == sympy.Symbol('x') + sympy.I * sympy.Symbol('y')