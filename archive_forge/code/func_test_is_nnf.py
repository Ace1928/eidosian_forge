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
def test_is_nnf():
    assert is_nnf(true) is True
    assert is_nnf(A) is True
    assert is_nnf(~A) is True
    assert is_nnf(A & B) is True
    assert is_nnf(A & B | ~A & A | ~B & B | ~A & ~B, False) is True
    assert is_nnf((A | B) & (~A | ~B)) is True
    assert is_nnf(Not(Or(A, B))) is False
    assert is_nnf(A ^ B) is False
    assert is_nnf(A & B | ~A & A | ~B & B | ~A & ~B, True) is False