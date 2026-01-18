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
def test_issue_14000():
    assert isinstance(sqrt(4, evaluate=False), Pow) == True
    assert isinstance(cbrt(3.5, evaluate=False), Pow) == True
    assert isinstance(root(16, 4, evaluate=False), Pow) == True
    assert sqrt(4, evaluate=False) == Pow(4, S.Half, evaluate=False)
    assert cbrt(3.5, evaluate=False) == Pow(3.5, Rational(1, 3), evaluate=False)
    assert root(4, 2, evaluate=False) == Pow(4, S.Half, evaluate=False)
    assert root(16, 4, 2, evaluate=False).has(Pow) == True
    assert real_root(-8, 3, evaluate=False).has(Pow) == True