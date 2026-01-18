import logging
from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP, warns_deprecated_sympy
import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.theanocode import (theano_code, dim_handling,
def test_Exp1():
    """
    Test that exp(1) prints without error and evaluates close to SymPy's E
    """
    e_a = sy.exp(1)
    e_b = sy.E
    np.testing.assert_allclose(float(e_a), np.e)
    np.testing.assert_allclose(float(e_b), np.e)
    e = theano_code_(e_a)
    np.testing.assert_allclose(float(e_a), e.eval())
    e = theano_code_(e_b)
    np.testing.assert_allclose(float(e_b), e.eval())