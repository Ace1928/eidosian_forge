import collections
from sympy.assumptions.ask import Q
from sympy.core.basic import (Basic, Atom, as_Basic,
from sympy.core.containers import Tuple
from sympy.core.function import Function, Lambda
from sympy.core.numbers import I, pi
from sympy.core.singleton import S
from sympy.core.symbol import symbols, Symbol, Dummy
from sympy.concrete.summations import Sum
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral
from sympy.functions.elementary.exponential import exp
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_literal_evalf_is_number_is_zero_is_comparable():
    x = symbols('x')
    f = Function('f')
    assert f.is_number is False
    assert f(1).is_number is False
    i = Integral(0, (x, x, x))
    assert i.n() == 0
    assert i.is_zero
    assert i.is_number is False
    assert i.evalf(2, strict=False) == 0
    n = sin(1) ** 2 + cos(1) ** 2 - 1
    assert n.is_comparable is False
    assert n.n(2).is_comparable is False
    assert n.n(2).n(2).is_comparable