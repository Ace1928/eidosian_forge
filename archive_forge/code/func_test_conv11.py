from symengine import (Symbol, Integer, sympify, SympifyError, log,
from symengine.lib.symengine_wrapper import (Subs, Derivative, RealMPFR,
import unittest
@unittest.skipIf(not have_sympy, 'SymPy not installed')
def test_conv11():
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    x1 = Symbol('x')
    y1 = Symbol('y')
    f = sympy.Function('f')
    f1 = Function('f')
    e1 = diff(f(2 * x, y), x)
    e2 = diff(f1(2 * x1, y1), x1)
    e3 = diff(f1(2 * x1, y1), y1)
    assert sympify(e1) == e2
    assert sympify(e1) != e3
    assert e2._sympy_() == e1
    assert e3._sympy_() != e1