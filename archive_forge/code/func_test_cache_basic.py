import logging
from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP, warns_deprecated_sympy
import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.theanocode import (theano_code, dim_handling,
def test_cache_basic():
    """ Test single symbol-like objects are cached when printed by themselves. """
    pairs = [(x, sy.Symbol('x')), (X, sy.MatrixSymbol('X', *X.shape)), (f_t, sy.Function('f')(sy.Symbol('t')))]
    for s1, s2 in pairs:
        cache = {}
        st = theano_code_(s1, cache=cache)
        assert theano_code_(s1, cache=cache) is st
        assert theano_code_(s1, cache={}) is not st
        assert theano_code_(s2, cache=cache) is st