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
def test_as_dummy():
    _0, _1 = symbols('_0 _1')
    assert ConditionSet(x, x < 1, Interval(y, oo)).as_dummy() == ConditionSet(_0, _0 < 1, Interval(y, oo))
    assert ConditionSet(x, x < 1, Interval(x, oo)).as_dummy() == ConditionSet(_0, _0 < 1, Interval(x, oo))
    assert ConditionSet(x, x < 1, ImageSet(Lambda(y, y ** 2), S.Integers)).as_dummy() == ConditionSet(_0, _0 < 1, ImageSet(Lambda(_0, _0 ** 2), S.Integers))
    e = ConditionSet((x, y), x <= y, S.Reals ** 2)
    assert e.bound_symbols == [x, y]
    assert e.as_dummy() == ConditionSet((_0, _1), _0 <= _1, S.Reals ** 2)
    assert e.as_dummy() == ConditionSet((y, x), y <= x, S.Reals ** 2).as_dummy()