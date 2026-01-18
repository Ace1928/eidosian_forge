from sympy.assumptions.ask import Q
from sympy.assumptions.refine import refine
from sympy.core.numbers import oo
from sympy.core.relational import Equality, Eq, Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions import Piecewise
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.sets.sets import (Interval, Union)
from sympy.simplify.simplify import simplify
from sympy.logic.boolalg import (
from sympy.assumptions.cnf import CNF
from sympy.testing.pytest import raises, XFAIL, slow
from itertools import combinations, permutations, product
def test_issue_8777():
    assert And(x > 2, x < oo).as_set() == Interval(2, oo, left_open=True)
    assert And(x >= 1, x < oo).as_set() == Interval(1, oo)
    assert (x < oo).as_set() == Interval(-oo, oo)
    assert (x > -oo).as_set() == Interval(-oo, oo)