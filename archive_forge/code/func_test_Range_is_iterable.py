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
def test_Range_is_iterable():
    assert Range(-100, 100).is_iterable is True
    assert Range(2, oo).is_iterable is False
    assert Range(-oo, 50).is_iterable is False
    assert Range(-oo, oo).is_iterable is False
    assert Range(oo, -oo).is_iterable is True
    assert Range(0, 0).is_iterable is True
    assert Range(oo, oo).is_iterable is True
    assert Range(-oo, -oo).is_iterable is True
    n = Symbol('n', integer=True)
    m = Symbol('m', integer=True)
    p = Symbol('p', integer=True, positive=True)
    assert Range(n, n + 49).is_iterable is True
    assert Range(n, 0).is_iterable is False
    assert Range(-3, n + 7).is_iterable is False
    assert Range(-3, p + 7).is_iterable is False
    assert Range(n, m).is_iterable is False
    assert Range(n + m, m - n).is_iterable is False
    assert Range(n, n + m + n).is_iterable is False
    assert Range(n, oo).is_iterable is False
    assert Range(-oo, n).is_iterable is False
    x = Symbol('x')
    assert Range(x, x + 49).is_iterable is False
    assert Range(x, 0).is_iterable is False
    assert Range(-3, x + 7).is_iterable is False
    assert Range(x, m).is_iterable is False
    assert Range(x + m, m - x).is_iterable is False
    assert Range(x, x + m + x).is_iterable is False
    assert Range(x, oo).is_iterable is False
    assert Range(-oo, x).is_iterable is False