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
def test_contains_1():
    assert Order(x).contains(Order(x))
    assert Order(x).contains(Order(x ** 2))
    assert not Order(x ** 2).contains(Order(x))
    assert not Order(x).contains(Order(1 / x))
    assert not Order(1 / x).contains(Order(exp(1 / x)))
    assert not Order(x).contains(Order(exp(1 / x)))
    assert Order(1 / x).contains(Order(x))
    assert Order(exp(1 / x)).contains(Order(x))
    assert Order(exp(1 / x)).contains(Order(1 / x))
    assert Order(exp(1 / x)).contains(Order(exp(1 / x)))
    assert Order(exp(2 / x)).contains(Order(exp(1 / x)))
    assert not Order(exp(1 / x)).contains(Order(exp(2 / x)))