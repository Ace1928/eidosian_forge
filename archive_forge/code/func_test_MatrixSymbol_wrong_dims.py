import logging
from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP, warns_deprecated_sympy
import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.theanocode import (theano_code, dim_handling,
@SKIP
def test_MatrixSymbol_wrong_dims():
    """ Test MatrixSymbol with invalid broadcastable. """
    bcs = [(), (False,), (True,), (True, False), (False, True), (True, True)]
    for bc in bcs:
        with raises(ValueError):
            theano_code_(X, broadcastables={X: bc})