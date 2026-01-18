import threading
from sympy.core.function import Function, UndefinedFunction
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import (GreaterThan, LessThan, StrictGreaterThan, StrictLessThan)
from sympy.core.symbol import (Dummy, Symbol, Wild, symbols)
from sympy.core.sympify import sympify  # can't import as S yet
from sympy.core.symbol import uniquely_named_symbol, _symbol, Str
from sympy.testing.pytest import raises, skip_under_pyodide
from sympy.core.symbol import disambiguate
def test_lt_gt():
    S = sympify
    x, y = (Symbol('x'), Symbol('y'))
    assert (x >= y) == GreaterThan(x, y)
    assert (x >= 0) == GreaterThan(x, 0)
    assert (x <= y) == LessThan(x, y)
    assert (x <= 0) == LessThan(x, 0)
    assert (0 <= x) == GreaterThan(x, 0)
    assert (0 >= x) == LessThan(x, 0)
    assert (S(0) >= x) == GreaterThan(0, x)
    assert (S(0) <= x) == LessThan(0, x)
    assert (x > y) == StrictGreaterThan(x, y)
    assert (x > 0) == StrictGreaterThan(x, 0)
    assert (x < y) == StrictLessThan(x, y)
    assert (x < 0) == StrictLessThan(x, 0)
    assert (0 < x) == StrictGreaterThan(x, 0)
    assert (0 > x) == StrictLessThan(x, 0)
    assert (S(0) > x) == StrictGreaterThan(0, x)
    assert (S(0) < x) == StrictLessThan(0, x)
    e = x ** 2 + 4 * x + 1
    assert (e >= 0) == GreaterThan(e, 0)
    assert (0 <= e) == GreaterThan(e, 0)
    assert (e > 0) == StrictGreaterThan(e, 0)
    assert (0 < e) == StrictGreaterThan(e, 0)
    assert (e <= 0) == LessThan(e, 0)
    assert (0 >= e) == LessThan(e, 0)
    assert (e < 0) == StrictLessThan(e, 0)
    assert (0 > e) == StrictLessThan(e, 0)
    assert (S(0) >= e) == GreaterThan(0, e)
    assert (S(0) <= e) == LessThan(0, e)
    assert (S(0) < e) == StrictLessThan(0, e)
    assert (S(0) > e) == StrictGreaterThan(0, e)