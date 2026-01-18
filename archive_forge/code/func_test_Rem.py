import itertools as it
from sympy.core.expr import unchanged
from sympy.core.function import Function
from sympy.core.numbers import I, oo, Rational
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.external import import_module
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.integers import floor, ceiling
from sympy.functions.elementary.miscellaneous import (sqrt, cbrt, root, Min,
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.functions.special.delta_functions import Heaviside
from sympy.utilities.lambdify import lambdify
from sympy.testing.pytest import raises, skip, ignore_warnings
def test_Rem():
    from sympy.abc import x, y
    assert Rem(5, 3) == 2
    assert Rem(-5, 3) == -2
    assert Rem(5, -3) == 2
    assert Rem(-5, -3) == -2
    assert Rem(x ** 3, y) == Rem(x ** 3, y)
    assert Rem(Rem(-5, 3) + 3, 3) == 1