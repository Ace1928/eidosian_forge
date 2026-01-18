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
def test_matches_basic():
    instances = [Basic(b1, b1, b2), Basic(b1, b2, b1), Basic(b2, b1, b1), Basic(b1, b2), Basic(b2, b1), b2, b1]
    for i, b_i in enumerate(instances):
        for j, b_j in enumerate(instances):
            if i == j:
                assert b_i.matches(b_j) == {}
            else:
                assert b_i.matches(b_j) is None
    assert b1.match(b1) == {}