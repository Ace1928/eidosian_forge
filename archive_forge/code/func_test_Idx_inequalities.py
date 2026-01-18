from sympy.core import symbols, Symbol, Tuple, oo, Dummy
from sympy.tensor.indexed import IndexException
from sympy.testing.pytest import raises
from sympy.utilities.iterables import iterable
from sympy.concrete.summations import Sum
from sympy.core.function import Function, Subs, Derivative
from sympy.core.relational import (StrictLessThan, GreaterThan,
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.series.order import Order
from sympy.sets.fancysets import Range
from sympy.tensor.indexed import IndexedBase, Idx, Indexed
def test_Idx_inequalities():
    i14 = Idx('i14', (1, 4))
    i79 = Idx('i79', (7, 9))
    i46 = Idx('i46', (4, 6))
    i35 = Idx('i35', (3, 5))
    assert i14 <= 5
    assert i14 < 5
    assert not i14 >= 5
    assert not i14 > 5
    assert 5 >= i14
    assert 5 > i14
    assert not 5 <= i14
    assert not 5 < i14
    assert LessThan(i14, 5)
    assert StrictLessThan(i14, 5)
    assert not GreaterThan(i14, 5)
    assert not StrictGreaterThan(i14, 5)
    assert i14 <= 4
    assert isinstance(i14 < 4, StrictLessThan)
    assert isinstance(i14 >= 4, GreaterThan)
    assert not i14 > 4
    assert isinstance(i14 <= 1, LessThan)
    assert not i14 < 1
    assert i14 >= 1
    assert isinstance(i14 > 1, StrictGreaterThan)
    assert not i14 <= 0
    assert not i14 < 0
    assert i14 >= 0
    assert i14 > 0
    from sympy.abc import x
    assert isinstance(i14 < x, StrictLessThan)
    assert isinstance(i14 > x, StrictGreaterThan)
    assert isinstance(i14 <= x, LessThan)
    assert isinstance(i14 >= x, GreaterThan)
    assert i14 < i79
    assert i14 <= i79
    assert not i14 > i79
    assert not i14 >= i79
    assert i14 <= i46
    assert isinstance(i14 < i46, StrictLessThan)
    assert isinstance(i14 >= i46, GreaterThan)
    assert not i14 > i46
    assert isinstance(i14 < i35, StrictLessThan)
    assert isinstance(i14 > i35, StrictGreaterThan)
    assert isinstance(i14 <= i35, LessThan)
    assert isinstance(i14 >= i35, GreaterThan)
    iNone1 = Idx('iNone1')
    iNone2 = Idx('iNone2')
    assert isinstance(iNone1 < iNone2, StrictLessThan)
    assert isinstance(iNone1 > iNone2, StrictGreaterThan)
    assert isinstance(iNone1 <= iNone2, LessThan)
    assert isinstance(iNone1 >= iNone2, GreaterThan)