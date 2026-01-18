from sympy.core.logic import fuzzy_and
from sympy.core.sympify import _sympify
from sympy.multipledispatch import dispatch
from sympy.testing.pytest import XFAIL, raises
from sympy.assumptions.ask import Q
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.integers import (ceiling, floor)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.logic.boolalg import (And, Implies, Not, Or, Xor)
from sympy.sets import Reals
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
from sympy.core.relational import (Relational, Equality, Unequality,
from sympy.sets.sets import Interval, FiniteSet
from itertools import combinations
def test_infinite_symbol_inequalities():
    x = Symbol('x', extended_positive=True, infinite=True)
    y = Symbol('y', extended_positive=True, infinite=True)
    z = Symbol('z', extended_negative=True, infinite=True)
    w = Symbol('w', extended_negative=True, infinite=True)
    inf_set = (x, y, oo)
    ninf_set = (z, w, -oo)
    for inf1 in inf_set:
        assert (inf1 < 1) is S.false
        assert (inf1 > 1) is S.true
        assert (inf1 <= 1) is S.false
        assert (inf1 >= 1) is S.true
        for inf2 in inf_set:
            assert (inf1 < inf2) is S.false
            assert (inf1 > inf2) is S.false
            assert (inf1 <= inf2) is S.true
            assert (inf1 >= inf2) is S.true
        for ninf1 in ninf_set:
            assert (inf1 < ninf1) is S.false
            assert (inf1 > ninf1) is S.true
            assert (inf1 <= ninf1) is S.false
            assert (inf1 >= ninf1) is S.true
            assert (ninf1 < inf1) is S.true
            assert (ninf1 > inf1) is S.false
            assert (ninf1 <= inf1) is S.true
            assert (ninf1 >= inf1) is S.false
    for ninf1 in ninf_set:
        assert (ninf1 < 1) is S.true
        assert (ninf1 > 1) is S.false
        assert (ninf1 <= 1) is S.true
        assert (ninf1 >= 1) is S.false
        for ninf2 in ninf_set:
            assert (ninf1 < ninf2) is S.false
            assert (ninf1 > ninf2) is S.false
            assert (ninf1 <= ninf2) is S.true
            assert (ninf1 >= ninf2) is S.true