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
def test_multivar_3():
    assert (Order(x) + Order(y)).args in [(Order(x), Order(y)), (Order(y), Order(x))]
    assert Order(x) + Order(y) + Order(x + y) == Order(x + y)
    assert (Order(x ** 2 * y) + Order(y ** 2 * x)).args in [(Order(x * y ** 2), Order(y * x ** 2)), (Order(y * x ** 2), Order(x * y ** 2))]
    assert Order(x ** 2 * y) + Order(y * x) == Order(x * y)