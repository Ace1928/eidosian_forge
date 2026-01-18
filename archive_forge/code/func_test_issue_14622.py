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
def test_issue_14622():
    assert (x ** (-4) + x ** (-3) + x ** (-1) + O(x ** (-6), (x, oo))).as_numer_denom() == (x ** 4 + x ** 5 + x ** 7 + O(x ** 2, (x, oo)), x ** 8)
    assert (x ** 3 + O(x ** 2, (x, oo))).is_Add
    assert O(x ** 2, (x, oo)).contains(x ** 3) is False
    assert O(x, (x, oo)).contains(O(x, (x, 0))) is None
    assert O(x, (x, 0)).contains(O(x, (x, oo))) is None
    raises(NotImplementedError, lambda: O(x ** 3).contains(x ** w))