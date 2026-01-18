import logging
from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP, warns_deprecated_sympy
import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.theanocode import (theano_code, dim_handling,
def test_many():
    """ Test printing a complex expression with multiple symbols. """
    expr = sy.exp(x ** 2 + sy.cos(y)) * sy.log(2 * z)
    comp = theano_code_(expr)
    expected = tt.exp(xt ** 2 + tt.cos(yt)) * tt.log(2 * zt)
    assert theq(comp, expected)