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
def test_ComplexRegion_contains():
    r = Symbol('r', real=True)
    a = Interval(2, 3)
    b = Interval(4, 6)
    c = Interval(7, 9)
    c1 = ComplexRegion(a * b)
    c2 = ComplexRegion(Union(a * b, c * a))
    assert 2.5 + 4.5 * I in c1
    assert 2 + 4 * I in c1
    assert 3 + 4 * I in c1
    assert 8 + 2.5 * I in c2
    assert 2.5 + 6.1 * I not in c1
    assert 4.5 + 3.2 * I not in c1
    assert c1.contains(x) == Contains(x, c1, evaluate=False)
    assert c1.contains(r) == False
    assert c2.contains(x) == Contains(x, c2, evaluate=False)
    assert c2.contains(r) == False
    r1 = Interval(0, 1)
    theta1 = Interval(0, 2 * S.Pi)
    c3 = ComplexRegion(r1 * theta1, polar=True)
    assert 0.5 + I * 6 / 10 in c3
    assert S.Half + I * 6 / 10 in c3
    assert S.Half + 0.6 * I in c3
    assert 0.5 + 0.6 * I in c3
    assert I in c3
    assert 1 in c3
    assert 0 in c3
    assert 1 + I not in c3
    assert 1 - I not in c3
    assert c3.contains(x) == Contains(x, c3, evaluate=False)
    assert c3.contains(r + 2 * I) == Contains(r + 2 * I, c3, evaluate=False)
    assert c3.contains(1 / (1 + r ** 2)) == Contains(1 / (1 + r ** 2), c3, evaluate=False)
    r2 = Interval(0, 3)
    theta2 = Interval(pi, 2 * pi, left_open=True)
    c4 = ComplexRegion(r2 * theta2, polar=True)
    assert c4.contains(0) == True
    assert c4.contains(2 + I) == False
    assert c4.contains(-2 + I) == False
    assert c4.contains(-2 - I) == True
    assert c4.contains(2 - I) == True
    assert c4.contains(-2) == False
    assert c4.contains(2) == True
    assert c4.contains(x) == Contains(x, c4, evaluate=False)
    assert c4.contains(3 / (1 + r ** 2)) == Contains(3 / (1 + r ** 2), c4, evaluate=False)
    raises(ValueError, lambda: ComplexRegion(r1 * theta1, polar=2))