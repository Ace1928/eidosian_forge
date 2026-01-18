from symengine import (Symbol, Integer, sympify, SympifyError, log,
from symengine.lib.symengine_wrapper import (Subs, Derivative, RealMPFR,
import unittest
@unittest.skipIf(not have_sympy, 'SymPy not installed')
def test_conv_doubles():
    f = 4.347249999999999
    a = sympify(f)
    assert isinstance(a, RealDouble)
    assert sympify(a._sympy_()) == a
    assert float(a) == f
    assert float(a._sympy_()) == f