from sympy.core.expr import unchanged
from sympy.sets.contains import Contains
from sympy.sets.fancysets import (ImageSet, Range, normalize_theta_set,
from sympy.sets.sets import (FiniteSet, Interval, Union, imageset,
from sympy.sets.conditionset import ConditionSet
from sympy.simplify.simplify import simplify
from sympy.core.basic import Basic
from sympy.core.containers import Tuple, TupleKind
from sympy.core.function import Lambda
from sympy.core.kind import NumberKind
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.logic.boolalg import And
from sympy.matrices.dense import eye
from sympy.testing.pytest import XFAIL, raises
from sympy.abc import x, y, t, z
from sympy.core.mod import Mod
import itertools
def test_Rationals():
    assert S.Integers.is_subset(S.Rationals)
    assert S.Naturals.is_subset(S.Rationals)
    assert S.Naturals0.is_subset(S.Rationals)
    assert S.Rationals.is_subset(S.Reals)
    assert S.Rationals.inf is -oo
    assert S.Rationals.sup is oo
    it = iter(S.Rationals)
    assert [next(it) for i in range(12)] == [0, 1, -1, S.Half, 2, Rational(-1, 2), -2, Rational(1, 3), 3, Rational(-1, 3), -3, Rational(2, 3)]
    assert Basic() not in S.Rationals
    assert S.Half in S.Rationals
    assert S.Rationals.contains(0.5) == Contains(0.5, S.Rationals, evaluate=False)
    assert 2 in S.Rationals
    r = symbols('r', rational=True)
    assert r in S.Rationals
    raises(TypeError, lambda: x in S.Rationals)
    assert S.Rationals.boundary == S.Reals
    assert S.Rationals.closure == S.Reals
    assert S.Rationals.is_open == False
    assert S.Rationals.is_closed == False