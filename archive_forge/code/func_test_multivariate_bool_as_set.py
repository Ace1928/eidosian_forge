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
@XFAIL
def test_multivariate_bool_as_set():
    x, y = symbols('x,y')
    assert And(x >= 0, y >= 0).as_set() == Interval(0, oo) * Interval(0, oo)
    assert Or(x >= 0, y >= 0).as_set() == S.Reals * S.Reals - Interval(-oo, 0, True, True) * Interval(-oo, 0, True, True)