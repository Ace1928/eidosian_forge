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
def test_leading_order2():
    assert set((2 + pi + x ** 2).extract_leading_order(x)) == {(pi, O(1, x)), (S(2), O(1, x))}
    assert set((2 * x + pi * x + x ** 2).extract_leading_order(x)) == {(2 * x, O(x)), (x * pi, O(x))}