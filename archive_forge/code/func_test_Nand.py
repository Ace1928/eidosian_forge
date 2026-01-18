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
def test_Nand():
    assert Nand() is false
    assert Nand(A) == ~A
    assert Nand(True) is false
    assert Nand(False) is true
    assert Nand(True, True) is false
    assert Nand(True, False) is true
    assert Nand(False, False) is true
    assert Nand(True, A) == ~A
    assert Nand(False, A) is true
    assert Nand(True, True, True) is false
    assert Nand(True, True, A) == ~A
    assert Nand(True, False, A) is true