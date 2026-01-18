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
def test_issue_12533():
    d = IndexedBase('d')
    assert IndexedBase(range(5)) == Range(0, 5, 1)
    assert d[0].subs(Symbol('d'), range(5)) == 0
    assert d[0].subs(d, range(5)) == 0
    assert d[1].subs(d, range(5)) == 1
    assert Indexed(Range(5), 2) == 2