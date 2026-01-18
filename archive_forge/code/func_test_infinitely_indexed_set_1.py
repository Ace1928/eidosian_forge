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
def test_infinitely_indexed_set_1():
    from sympy.abc import n, m
    assert imageset(Lambda(n, n), S.Integers) == imageset(Lambda(m, m), S.Integers)
    assert imageset(Lambda(n, 2 * n), S.Integers).intersect(imageset(Lambda(m, 2 * m + 1), S.Integers)) is S.EmptySet
    assert imageset(Lambda(n, 2 * n), S.Integers).intersect(imageset(Lambda(n, 2 * n + 1), S.Integers)) is S.EmptySet
    assert imageset(Lambda(m, 2 * m), S.Integers).intersect(imageset(Lambda(n, 3 * n), S.Integers)).dummy_eq(ImageSet(Lambda(t, 6 * t), S.Integers))
    assert imageset(x, x / 2 + Rational(1, 3), S.Integers).intersect(S.Integers) is S.EmptySet
    assert imageset(x, x / 2 + S.Half, S.Integers).intersect(S.Integers) is S.Integers
    S53 = ImageSet(Lambda(n, 5 * n + 3), S.Integers)
    assert S53.intersect(S.Integers) == S53