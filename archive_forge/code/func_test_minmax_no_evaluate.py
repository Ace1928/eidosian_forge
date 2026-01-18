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
def test_minmax_no_evaluate():
    from sympy import evaluate
    p = Symbol('p', positive=True)
    assert Max(1, 3) == 3
    assert Max(1, 3).args == ()
    assert Max(0, p) == p
    assert Max(0, p).args == ()
    assert Min(0, p) == 0
    assert Min(0, p).args == ()
    assert Max(1, 3, evaluate=False) != 3
    assert Max(1, 3, evaluate=False).args == (1, 3)
    assert Max(0, p, evaluate=False) != p
    assert Max(0, p, evaluate=False).args == (0, p)
    assert Min(0, p, evaluate=False) != 0
    assert Min(0, p, evaluate=False).args == (0, p)
    with evaluate(False):
        assert Max(1, 3) != 3
        assert Max(1, 3).args == (1, 3)
        assert Max(0, p) != p
        assert Max(0, p).args == (0, p)
        assert Min(0, p) != 0
        assert Min(0, p).args == (0, p)