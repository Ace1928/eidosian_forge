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
def test_ComplexRegion_union():
    c1 = ComplexRegion(Interval(0, 1) * Interval(0, 2 * S.Pi), polar=True)
    c2 = ComplexRegion(Interval(0, 1) * Interval(0, S.Pi), polar=True)
    c3 = ComplexRegion(Interval(0, oo) * Interval(0, S.Pi), polar=True)
    c4 = ComplexRegion(Interval(0, oo) * Interval(S.Pi, 2 * S.Pi), polar=True)
    p1 = Union(Interval(0, 1) * Interval(0, 2 * S.Pi), Interval(0, 1) * Interval(0, S.Pi))
    p2 = Union(Interval(0, oo) * Interval(0, S.Pi), Interval(0, oo) * Interval(S.Pi, 2 * S.Pi))
    assert c1.union(c2) == ComplexRegion(p1, polar=True)
    assert c3.union(c4) == ComplexRegion(p2, polar=True)
    c5 = ComplexRegion(Interval(2, 5) * Interval(6, 9))
    c6 = ComplexRegion(Interval(4, 6) * Interval(10, 12))
    c7 = ComplexRegion(Interval(0, 10) * Interval(-10, 0))
    c8 = ComplexRegion(Interval(12, 16) * Interval(14, 20))
    p3 = Union(Interval(2, 5) * Interval(6, 9), Interval(4, 6) * Interval(10, 12))
    p4 = Union(Interval(0, 10) * Interval(-10, 0), Interval(12, 16) * Interval(14, 20))
    assert c5.union(c6) == ComplexRegion(p3)
    assert c7.union(c8) == ComplexRegion(p4)
    assert c1.union(Interval(2, 4)) == Union(c1, Interval(2, 4), evaluate=False)
    assert c5.union(Interval(2, 4)) == Union(c5, ComplexRegion.from_real(Interval(2, 4)))