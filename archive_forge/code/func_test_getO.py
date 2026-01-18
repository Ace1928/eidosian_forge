from sympy.core.add import Add
from sympy.core.function import (Function, expand)
from sympy.core.numbers import (I, Rational, nan, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (conjugate, transpose)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.series.order import O, Order
from sympy.core.expr import unchanged
from sympy.testing.pytest import raises
from sympy.abc import w, x, y, z
def test_getO():
    assert x.getO() is None
    assert x.removeO() == x
    assert O(x).getO() == O(x)
    assert O(x).removeO() == 0
    assert (z + O(x) + O(y)).getO() == O(x) + O(y)
    assert (z + O(x) + O(y)).removeO() == z
    raises(NotImplementedError, lambda: (O(x) + O(y)).getn())