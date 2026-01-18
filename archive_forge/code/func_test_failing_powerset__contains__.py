from sympy.core.expr import unchanged
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Interval
from sympy.sets.powerset import PowerSet
from sympy.sets.sets import FiniteSet
from sympy.testing.pytest import raises, XFAIL
@XFAIL
def test_failing_powerset__contains__():
    assert FiniteSet(1, 2) not in PowerSet(S.EmptySet).rewrite(FiniteSet)
    assert S.Naturals not in PowerSet(S.EmptySet).rewrite(FiniteSet)
    assert S.Naturals not in PowerSet(FiniteSet(1, 2)).rewrite(FiniteSet)
    assert S.Naturals0 not in PowerSet(S.EmptySet).rewrite(FiniteSet)
    assert S.Naturals0 not in PowerSet(FiniteSet(1, 2)).rewrite(FiniteSet)
    assert S.Integers not in PowerSet(S.EmptySet).rewrite(FiniteSet)
    assert S.Integers not in PowerSet(FiniteSet(1, 2)).rewrite(FiniteSet)
    assert S.Rationals not in PowerSet(S.EmptySet).rewrite(FiniteSet)
    assert S.Rationals not in PowerSet(FiniteSet(1, 2)).rewrite(FiniteSet)
    assert S.Reals not in PowerSet(S.EmptySet).rewrite(FiniteSet)
    assert S.Reals not in PowerSet(FiniteSet(1, 2)).rewrite(FiniteSet)
    assert S.Complexes not in PowerSet(S.EmptySet).rewrite(FiniteSet)
    assert S.Complexes not in PowerSet(FiniteSet(1, 2)).rewrite(FiniteSet)