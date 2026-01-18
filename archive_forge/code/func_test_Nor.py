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
def test_Nor():
    assert Nor() is true
    assert Nor(A) == ~A
    assert Nor(True) is false
    assert Nor(False) is true
    assert Nor(True, True) is false
    assert Nor(True, False) is false
    assert Nor(False, False) is true
    assert Nor(True, A) is false
    assert Nor(False, A) == ~A
    assert Nor(True, True, True) is false
    assert Nor(True, True, A) is false
    assert Nor(True, False, A) is false