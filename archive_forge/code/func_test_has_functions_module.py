from symengine.sympy_compat import (Integer, Rational, S, Basic, Add, Mul,
from symengine.test_utilities import raises
def test_has_functions_module():
    import symengine.sympy_compat as sp
    assert sp.functions.sin(0) == 0