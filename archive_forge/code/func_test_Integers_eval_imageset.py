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
def test_Integers_eval_imageset():
    ans = ImageSet(Lambda(x, 2 * x + Rational(3, 7)), S.Integers)
    im = imageset(Lambda(x, -2 * x + Rational(3, 7)), S.Integers)
    assert im == ans
    im = imageset(Lambda(x, -2 * x - Rational(11, 7)), S.Integers)
    assert im == ans
    y = Symbol('y')
    L = imageset(x, 2 * x + y, S.Integers)
    assert y + 4 in L
    a, b, c = (0.092, 0.433, 0.341)
    assert a in imageset(x, a + c * x, S.Integers)
    assert b in imageset(x, b + c * x, S.Integers)
    _x = symbols('x', negative=True)
    eq = _x ** 2 - _x + 1
    assert imageset(_x, eq, S.Integers).lamda.expr == _x ** 2 + _x + 1
    eq = 3 * _x - 1
    assert imageset(_x, eq, S.Integers).lamda.expr == 3 * _x + 2
    assert imageset(x, (x, 1 / x), S.Integers) == ImageSet(Lambda(x, (x, 1 / x)), S.Integers)