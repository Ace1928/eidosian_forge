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
def test_inequalities_symbol_name_same():
    """Using the operator and functional forms should give same results."""
    A = (x, y, S.Zero, S.One / 3, pi, oo, -oo)
    for a in A:
        for b in A:
            assert Gt(a, b) == (a > b)
            assert Lt(a, b) == (a < b)
            assert Ge(a, b) == (a >= b)
            assert Le(a, b) == (a <= b)
    for b in (y, S.Zero, S.One / 3, pi, oo, -oo):
        assert Gt(x, b, evaluate=False) == (x > b)
        assert Lt(x, b, evaluate=False) == (x < b)
        assert Ge(x, b, evaluate=False) == (x >= b)
        assert Le(x, b, evaluate=False) == (x <= b)
    for b in (y, S.Zero, S.One / 3, pi, oo, -oo):
        assert Gt(b, x, evaluate=False) == (b > x)
        assert Lt(b, x, evaluate=False) == (b < x)
        assert Ge(b, x, evaluate=False) == (b >= x)
        assert Le(b, x, evaluate=False) == (b <= x)