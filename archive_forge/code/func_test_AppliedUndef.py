import logging
from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP, warns_deprecated_sympy
import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.theanocode import (theano_code, dim_handling,
def test_AppliedUndef():
    """ Test printing AppliedUndef instance, which works similarly to Symbol. """
    ftt = theano_code_(f_t)
    assert isinstance(ftt, tt.TensorVariable)
    assert ftt.broadcastable == ()
    assert ftt.name == 'f_t'