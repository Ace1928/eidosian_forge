from sympy.core.expr import unchanged
from sympy.sets import (ConditionSet, Intersection, FiniteSet,
from sympy.sets.sets import SetKind
from sympy.core.function import (Function, Lambda)
from sympy.core.mod import Mod
from sympy.core.kind import NumberKind
from sympy.core.numbers import (oo, pi)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.trigonometric import (asin, sin)
from sympy.logic.boolalg import And
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.sets.sets import Interval
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_dummy_eq():
    C = ConditionSet
    I = S.Integers
    c = C(x, x < 1, I)
    assert c.dummy_eq(C(y, y < 1, I))
    assert c.dummy_eq(1) == False
    assert c.dummy_eq(C(x, x < 1, S.Reals)) == False
    c1 = ConditionSet((x, y), Eq(x + 1, 0) & Eq(x + y, 0), S.Reals ** 2)
    c2 = ConditionSet((x, y), Eq(x + 1, 0) & Eq(x + y, 0), S.Reals ** 2)
    c3 = ConditionSet((x, y), Eq(x + 1, 0) & Eq(x + y, 0), S.Complexes ** 2)
    assert c1.dummy_eq(c2)
    assert c1.dummy_eq(c3) is False
    assert c.dummy_eq(c1) is False
    assert c1.dummy_eq(c) is False
    m = Symbol('m')
    n = Symbol('n')
    a = Symbol('a')
    d1 = ImageSet(Lambda(m, m * pi), S.Integers)
    d2 = ImageSet(Lambda(n, n * pi), S.Integers)
    c1 = ConditionSet(x, Ne(a, 0), d1)
    c2 = ConditionSet(x, Ne(a, 0), d2)
    assert c1.dummy_eq(c2)