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
def test_issue_8373():
    x = symbols('x', real=True)
    assert Or(x < 1, x > -1).simplify() == S.true
    assert Or(x < 1, x >= 1).simplify() == S.true
    assert And(x < 1, x >= 1).simplify() == S.false
    assert Or(x <= 1, x >= 1).simplify() == S.true