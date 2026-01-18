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
def test_CondSet():
    sin_sols_principal = ConditionSet(x, Eq(sin(x), 0), Interval(0, 2 * pi, False, True))
    assert pi in sin_sols_principal
    assert pi / 2 not in sin_sols_principal
    assert 3 * pi not in sin_sols_principal
    assert oo not in sin_sols_principal
    assert 5 in ConditionSet(x, x ** 2 > 4, S.Reals)
    assert 1 not in ConditionSet(x, x ** 2 > 4, S.Reals)
    assert 0 not in ConditionSet(x, y > 5, Interval(1, 7))
    raises(TypeError, lambda: 6 in ConditionSet(x, y > 5, Interval(1, 7)))
    X = MatrixSymbol('X', 2, 2)
    matrix_set = ConditionSet(X, Eq(X * Matrix([[1, 1], [1, 1]]), X))
    Y = Matrix([[0, 0], [0, 0]])
    assert matrix_set.contains(Y).doit() is S.true
    Z = Matrix([[1, 2], [3, 4]])
    assert matrix_set.contains(Z).doit() is S.false
    assert isinstance(ConditionSet(x, x < 1, {x, y}).base_set, FiniteSet)
    raises(TypeError, lambda: ConditionSet(x, x + 1, {x, y}))
    raises(TypeError, lambda: ConditionSet(x, x, 1))
    I = S.Integers
    U = S.UniversalSet
    C = ConditionSet
    assert C(x, False, I) is S.EmptySet
    assert C(x, True, I) is I
    assert C(x, x < 1, C(x, x < 2, I)) == C(x, (x < 1) & (x < 2), I)
    assert C(y, y < 1, C(x, y < 2, I)) == C(x, (x < 1) & (y < 2), I), C(y, y < 1, C(x, y < 2, I))
    assert C(y, y < 1, C(x, x < 2, I)) == C(y, (y < 1) & (y < 2), I)
    assert C(y, y < 1, C(x, y < x, I)) == C(x, (x < 1) & (y < x), I)
    assert unchanged(C, y, x < 1, C(x, y < x, I))
    assert ConditionSet(x, x < 1).base_set is U
    assert ConditionSet((x,), x < 1).base_set is U
    c = ConditionSet((x, y), x < y, I ** 2)
    assert (1, 2) in c
    assert (1, pi) not in c
    raises(TypeError, lambda: C(x, x > 1, C((x, y), x > 1, I ** 2)))
    raises(TypeError, lambda: C((x, y), x + y < 2, U, U))