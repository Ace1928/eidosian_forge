import logging
from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP, warns_deprecated_sympy
import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.theanocode import (theano_code, dim_handling,
def test_Relationals():
    assert theq(theano_code_(sy.Eq(x, y)), tt.eq(xt, yt))
    assert theq(theano_code_(x > y), xt > yt)
    assert theq(theano_code_(x < y), xt < yt)
    assert theq(theano_code_(x >= y), xt >= yt)
    assert theq(theano_code_(x <= y), xt <= yt)