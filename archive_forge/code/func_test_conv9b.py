from symengine import (Symbol, Integer, sympify, SympifyError, log,
from symengine.lib.symengine_wrapper import (Subs, Derivative, RealMPFR,
import unittest
@unittest.skipIf(not have_sympy, 'SymPy not installed')
def test_conv9b():
    x = Symbol('x')
    y = Symbol('y')
    assert sympify(sympy.I) == I
    assert sympify(2 * sympy.I + 3) == 2 * I + 3
    assert sympify(2 * sympy.I / 5 + sympy.S(3) / 5) == 2 * I / 5 + Integer(3) / 5
    assert sympify(sympy.Symbol('x') * sympy.I + 3) == x * I + 3
    assert sympify(sympy.Symbol('x') + sympy.I * sympy.Symbol('y')) == x + I * y