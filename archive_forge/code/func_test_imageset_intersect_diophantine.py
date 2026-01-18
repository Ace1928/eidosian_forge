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
def test_imageset_intersect_diophantine():
    from sympy.abc import m, n
    img1 = ImageSet(Lambda(n, 2 * n + 1), S.Integers)
    img2 = ImageSet(Lambda(n, 4 * n + 1), S.Integers)
    assert img1.intersect(img2) == img2
    assert ImageSet(Lambda(n, 2 * n), S.Integers).intersect(ImageSet(Lambda(n, 2 * n + 1), S.Integers)) == S.EmptySet
    assert ImageSet(Lambda(n, 9 / n + 20 * n / 3), S.Integers).intersect(S.Integers) == FiniteSet(-61, -23, 23, 61)
    assert ImageSet(Lambda(n, (n - 2) ** 2), S.Integers).intersect(ImageSet(Lambda(n, -(n - 3) ** 2), S.Integers)) == FiniteSet(0)
    assert ImageSet(Lambda(n, n ** 2 + 5), S.Integers).intersect(ImageSet(Lambda(m, 2 * m), S.Integers)).dummy_eq(ImageSet(Lambda(n, 4 * n ** 2 + 4 * n + 6), S.Integers))
    assert ImageSet(Lambda(n, n ** 2 - 9), S.Integers).intersect(ImageSet(Lambda(m, -m ** 2), S.Integers)) == FiniteSet(-9, 0)
    assert ImageSet(Lambda(m, m ** 2 + 40), S.Integers).intersect(ImageSet(Lambda(n, 41 * n), S.Integers)).dummy_eq(Intersection(ImageSet(Lambda(m, m ** 2 + 40), S.Integers), ImageSet(Lambda(n, 41 * n), S.Integers)))
    assert ImageSet(Lambda(n, n ** 4 - 2 ** 4), S.Integers).intersect(ImageSet(Lambda(m, -m ** 4 + 3 ** 4), S.Integers)) == FiniteSet(0, 65)
    assert ImageSet(Lambda(n, pi / 12 + n * 5 * pi / 12), S.Integers).intersect(ImageSet(Lambda(n, 7 * pi / 12 + n * 11 * pi / 12), S.Integers)).dummy_eq(ImageSet(Lambda(n, 55 * pi * n / 12 + 17 * pi / 4), S.Integers))
    assert ImageSet(Lambda(n, n * log(2)), S.Integers).intersection(S.Integers).dummy_eq(Intersection(ImageSet(Lambda(n, n * log(2)), S.Integers), S.Integers))
    assert ImageSet(Lambda(n, n ** 3 + 1), S.Integers).intersect(ImageSet(Lambda(n, n ** 3), S.Integers)).dummy_eq(Intersection(ImageSet(Lambda(n, n ** 3 + 1), S.Integers), ImageSet(Lambda(n, n ** 3), S.Integers)))