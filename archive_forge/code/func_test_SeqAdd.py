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
def test_SeqAdd():
    per = SeqPer((1, 2, 3), (n, 0, oo))
    form = SeqFormula(n ** 2)
    per_bou = SeqPer((1, 2), (n, 1, 5))
    form_bou = SeqFormula(n ** 2, (6, 10))
    form_bou2 = SeqFormula(n ** 2, (1, 5))
    assert SeqAdd() == S.EmptySequence
    assert SeqAdd(S.EmptySequence) == S.EmptySequence
    assert SeqAdd(per) == per
    assert SeqAdd(per, S.EmptySequence) == per
    assert SeqAdd(per_bou, form_bou) == S.EmptySequence
    s = SeqAdd(per_bou, form_bou2, evaluate=False)
    assert s.args == (form_bou2, per_bou)
    assert s[:] == [2, 6, 10, 18, 26]
    assert list(s) == [2, 6, 10, 18, 26]
    assert isinstance(SeqAdd(per, per_bou, evaluate=False), SeqAdd)
    s1 = SeqAdd(per, per_bou)
    assert isinstance(s1, SeqPer)
    assert s1 == SeqPer((2, 4, 4, 3, 3, 5), (n, 1, 5))
    s2 = SeqAdd(form, form_bou)
    assert isinstance(s2, SeqFormula)
    assert s2 == SeqFormula(2 * n ** 2, (6, 10))
    assert SeqAdd(form, form_bou, per) == SeqAdd(per, SeqFormula(2 * n ** 2, (6, 10)))
    assert SeqAdd(form, SeqAdd(form_bou, per)) == SeqAdd(per, SeqFormula(2 * n ** 2, (6, 10)))
    assert SeqAdd(per, SeqAdd(form, form_bou), evaluate=False) == SeqAdd(per, SeqFormula(2 * n ** 2, (6, 10)))
    assert SeqAdd(SeqPer((1, 2), (n, 0, oo)), SeqPer((1, 2), (m, 0, oo))) == SeqPer((2, 4), (n, 0, oo))