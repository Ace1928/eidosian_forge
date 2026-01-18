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
def test_issue_11463():
    numpy = import_module('numpy')
    if not numpy:
        skip('numpy not installed.')
    x = Symbol('x')
    f = lambdify(x, real_root(log(x / (x - 2)), 3), 'numpy')
    with ignore_warnings(RuntimeWarning):
        assert f(numpy.array(-1)) < -1