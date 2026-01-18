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
def test_multivar_0():
    assert Order(x * y).expr == x * y
    assert Order(x * y ** 2).expr == x * y ** 2
    assert Order(x * y, x).expr == x
    assert Order(x * y ** 2, y).expr == y ** 2
    assert Order(x * y * z).expr == x * y * z
    assert Order(x / y).expr == x / y
    assert Order(x * exp(1 / y)).expr == x * exp(1 / y)
    assert Order(exp(x) * exp(1 / y)).expr == exp(x) * exp(1 / y)