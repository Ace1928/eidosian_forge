from sympy.core.containers import Tuple
from sympy.core.function import Function
from sympy.core.numbers import oo, Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols, Symbol
from sympy.functions.combinatorial.numbers import tribonacci, fibonacci
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.series import EmptySequence
from sympy.series.sequences import (SeqMul, SeqAdd, SeqPer, SeqFormula,
from sympy.sets.sets import Interval
from sympy.tensor.indexed import Indexed, Idx
from sympy.series.sequences import SeqExpr, SeqExprOp, RecursiveSeq
from sympy.testing.pytest import raises, slow
@slow
def test_find_linear_recurrence():
    assert sequence((0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55), (n, 0, 10)).find_linear_recurrence(11) == [1, 1]
    assert sequence((1, 2, 4, 7, 28, 128, 582, 2745, 13021, 61699, 292521, 1387138), (n, 0, 11)).find_linear_recurrence(12) == [5, -2, 6, -11]
    assert sequence(x * n ** 3 + y * n, (n, 0, oo)).find_linear_recurrence(10) == [4, -6, 4, -1]
    assert sequence(x ** n, (n, 0, 20)).find_linear_recurrence(21) == [x]
    assert sequence((1, 2, 3)).find_linear_recurrence(10, 5) == [0, 0, 1]
    assert sequence(((1 + sqrt(5)) / 2) ** n + (-(1 + sqrt(5)) / 2) ** (-n)).find_linear_recurrence(10) == [1, 1]
    assert sequence(x * ((1 + sqrt(5)) / 2) ** n + y * (-(1 + sqrt(5)) / 2) ** (-n), (n, 0, oo)).find_linear_recurrence(10) == [1, 1]
    assert sequence((1, 2, 3, 4, 6), (n, 0, 4)).find_linear_recurrence(5) == []
    assert sequence((2, 3, 4, 5, 6, 79), (n, 0, 5)).find_linear_recurrence(6, gfvar=x) == ([], None)
    assert sequence((2, 3, 4, 5, 8, 30), (n, 0, 5)).find_linear_recurrence(6, gfvar=x) == ([Rational(19, 2), -20, Rational(27, 2)], (-31 * x ** 2 + 32 * x - 4) / (27 * x ** 3 - 40 * x ** 2 + 19 * x - 2))
    assert sequence(fibonacci(n)).find_linear_recurrence(30, gfvar=x) == ([1, 1], -x / (x ** 2 + x - 1))
    assert sequence(tribonacci(n)).find_linear_recurrence(30, gfvar=x) == ([1, 1, 1], -x / (x ** 3 + x ** 2 + x - 1))