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
def test_issue_11732():
    interval12 = Interval(1, 2)
    finiteset1234 = FiniteSet(1, 2, 3, 4)
    pointComplex = Tuple(1, 5)
    assert (interval12 in S.Naturals) == False
    assert (interval12 in S.Naturals0) == False
    assert (interval12 in S.Integers) == False
    assert (interval12 in S.Complexes) == False
    assert (finiteset1234 in S.Naturals) == False
    assert (finiteset1234 in S.Naturals0) == False
    assert (finiteset1234 in S.Integers) == False
    assert (finiteset1234 in S.Complexes) == False
    assert (pointComplex in S.Naturals) == False
    assert (pointComplex in S.Naturals0) == False
    assert (pointComplex in S.Integers) == False
    assert (pointComplex in S.Complexes) == True