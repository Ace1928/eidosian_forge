import logging
from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP, warns_deprecated_sympy
import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.theanocode import (theano_code, dim_handling,
def test_cache_types_distinct():
    """
    Test that symbol-like objects of different types (Symbol, MatrixSymbol,
    AppliedUndef) are distinguished by the cache even if they have the same
    name.
    """
    symbols = [sy.Symbol('f_t'), sy.MatrixSymbol('f_t', 4, 4), f_t]
    cache = {}
    printed = {}
    for s in symbols:
        st = theano_code_(s, cache=cache)
        assert st not in printed.values()
        printed[s] = st
    assert len(set(map(id, printed.values()))) == len(symbols)
    for s, st in printed.items():
        with warns_deprecated_sympy():
            assert theano_code(s, cache=cache) is st