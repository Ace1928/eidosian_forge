from symengine import (Symbol, Integer, sympify, SympifyError, log,
from symengine.lib.symengine_wrapper import (Subs, Derivative, RealMPFR,
import unittest
@unittest.skipIf(not have_sympy, 'SymPy not installed')
def test_conv8():
    e1 = function_symbol('f', Symbol('x'))
    e2 = function_symbol('g', Symbol('x'), Symbol('y'))
    assert e1._sympy_() == sympy.Function('f')(sympy.Symbol('x'))
    assert e2._sympy_() != sympy.Function('f')(sympy.Symbol('x'))
    assert e2._sympy_() == sympy.Function('g')(sympy.Symbol('x'), sympy.Symbol('y'))
    e3 = function_symbol('q', Symbol('t'))
    assert e3._sympy_() == sympy.Function('q')(sympy.Symbol('t'))
    assert e3._sympy_() != sympy.Function('f')(sympy.Symbol('t'))
    assert e3._sympy_() != sympy.Function('q')(sympy.Symbol('t'), sympy.Symbol('t'))