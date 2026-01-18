from symengine import (Symbol, Integer, sympify, SympifyError, log,
from symengine.lib.symengine_wrapper import (Subs, Derivative, RealMPFR,
import unittest
@unittest.skipIf(not have_sympy, 'SymPy not installed')
def test_mpc():
    if have_mpc:
        a = ComplexMPC('1', '2', 100)
        b = sympy.Float(1, 29) + sympy.Float(2, 29) * sympy.I
        assert sympify(b) == a
        assert b == a._sympy_()