from symengine.test_utilities import raises
from symengine.lib.symengine_wrapper import (Interval, EmptySet, UniversalSet,
def test_ConditionSet():
    x = Symbol('x')
    i1 = Interval(-oo, oo)
    f1 = FiniteSet(0, 1, 2, 4)
    cond1 = Ge(x ** 2, 9)
    assert ConditionSet(x, And(Eq(0, 1), i1.contains(x))) == EmptySet()
    assert ConditionSet(x, And(Gt(1, 0), i1.contains(x))) == i1
    assert ConditionSet(x, And(cond1, f1.contains(x))) == FiniteSet(4)