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
def test_issue_18904():
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15 = symbols('x1:16')
    eq = x1 & x2 & x3 & x4 & x5 & x6 & x7 & x8 & x9 | x1 & x2 & x3 & x4 & x5 & x6 & x7 & x10 & x9 | x1 & x11 & x3 & x12 & x5 & x13 & x14 & x15 & x9
    assert is_cnf(to_cnf(eq))
    raises(ValueError, lambda: to_cnf(eq, simplify=True))
    for f, t in zip((And, Or), (to_cnf, to_dnf)):
        eq = f(x1, x2, x3, x4, x5, x6, x7, x8, x9)
        raises(ValueError, lambda: to_cnf(eq, simplify=True))
        assert t(eq, simplify=True, force=True) == eq