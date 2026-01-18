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
def test_Idx_limits():
    i = symbols('i', cls=Idx)
    r = Indexed('r', i)
    assert SeqFormula(r, (i, 0, 5))[:] == [r.subs(i, j) for j in range(6)]
    assert SeqPer((1, 2), (i, 0, 5))[:] == [1, 2, 1, 2, 1, 2]