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
def test_all_or_nothing():
    x = symbols('x', extended_real=True)
    args = (x >= -oo, x <= oo)
    v = And(*args)
    if v.func is And:
        assert len(v.args) == len(args) - args.count(S.true)
    else:
        assert v == True
    v = Or(*args)
    if v.func is Or:
        assert len(v.args) == 2
    else:
        assert v == True