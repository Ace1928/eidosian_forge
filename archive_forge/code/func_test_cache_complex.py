import logging
from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP, warns_deprecated_sympy
import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.theanocode import (theano_code, dim_handling,
def test_cache_complex():
    """
    Test caching on a complicated expression with multiple symbols appearing
    multiple times.
    """
    expr = x ** 2 + (y - sy.exp(x)) * sy.sin(z - x * y)
    symbol_names = {s.name for s in expr.free_symbols}
    expr_t = theano_code_(expr)
    seen = set()
    for v in theano.gof.graph.ancestors([expr_t]):
        if v.owner is None and (not isinstance(v, theano.gof.graph.Constant)):
            assert v.name in symbol_names
            assert v.name not in seen
            seen.add(v.name)
    assert seen == symbol_names