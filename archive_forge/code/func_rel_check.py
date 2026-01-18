from sympy.core.logic import fuzzy_and
from sympy.core.sympify import _sympify
from sympy.multipledispatch import dispatch
from sympy.testing.pytest import XFAIL, raises
from sympy.assumptions.ask import Q
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.integers import (ceiling, floor)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.logic.boolalg import (And, Implies, Not, Or, Xor)
from sympy.sets import Reals
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
from sympy.core.relational import (Relational, Equality, Unequality,
from sympy.sets.sets import Interval, FiniteSet
from itertools import combinations
def rel_check(a, b):
    from sympy.testing.pytest import raises
    assert a.is_number and b.is_number
    for do in range(len({type(a), type(b)})):
        if S.NaN in (a, b):
            v = [a == b, a != b]
            assert len(set(v)) == 1 and v[0] == False
            assert not a != b and (not a == b)
            assert raises(TypeError, lambda: a < b)
            assert raises(TypeError, lambda: a <= b)
            assert raises(TypeError, lambda: a > b)
            assert raises(TypeError, lambda: a >= b)
        else:
            E = [a == b, a != b]
            assert len(set(E)) == 2
            v = [a < b, a <= b, a > b, a >= b]
            i = [[True, True, False, False], [False, True, False, True], [False, False, True, True]].index(v)
            if i == 1:
                assert E[0] or a.is_Float != b.is_Float
            else:
                assert E[1]
        a, b = (b, a)
    return True