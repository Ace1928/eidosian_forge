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
def test_ANFform():
    x, y = symbols('x,y')
    assert ANFform([x], [1, 1]) == True
    assert ANFform([x], [0, 0]) == False
    assert ANFform([x], [1, 0]) == Xor(x, True, remove_true=False)
    assert ANFform([x, y], [1, 1, 1, 0]) == Xor(True, And(x, y), remove_true=False)