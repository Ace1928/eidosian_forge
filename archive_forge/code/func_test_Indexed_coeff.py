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
def test_Indexed_coeff():
    N = Symbol('N', integer=True)
    len_y = N
    i = Idx('i', len_y - 1)
    y = IndexedBase('y', shape=(len_y,))
    a = (1 / y[i + 1] * y[i]).coeff(y[i])
    b = (y[i] / y[i + 1]).coeff(y[i])
    assert a == b