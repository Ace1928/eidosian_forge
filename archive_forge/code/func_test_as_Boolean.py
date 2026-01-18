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
def test_as_Boolean():
    nz = symbols('nz', nonzero=True)
    assert all((as_Boolean(i) is S.true for i in (True, S.true, 1, nz)))
    z = symbols('z', zero=True)
    assert all((as_Boolean(i) is S.false for i in (False, S.false, 0, z)))
    assert all((as_Boolean(i) == i for i in (x, x < 0)))
    for i in (2, S(2), x + 1, []):
        raises(TypeError, lambda: as_Boolean(i))