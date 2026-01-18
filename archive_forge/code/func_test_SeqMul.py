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
def test_SeqMul():
    per = SeqPer((1, 2, 3), (n, 0, oo))
    form = SeqFormula(n ** 2)
    per_bou = SeqPer((1, 2), (n, 1, 5))
    form_bou = SeqFormula(n ** 2, (n, 6, 10))
    form_bou2 = SeqFormula(n ** 2, (1, 5))
    assert SeqMul() == S.EmptySequence
    assert SeqMul(S.EmptySequence) == S.EmptySequence
    assert SeqMul(per) == per
    assert SeqMul(per, S.EmptySequence) == S.EmptySequence
    assert SeqMul(per_bou, form_bou) == S.EmptySequence
    s = SeqMul(per_bou, form_bou2, evaluate=False)
    assert s.args == (form_bou2, per_bou)
    assert s[:] == [1, 8, 9, 32, 25]
    assert list(s) == [1, 8, 9, 32, 25]
    assert isinstance(SeqMul(per, per_bou, evaluate=False), SeqMul)
    s1 = SeqMul(per, per_bou)
    assert isinstance(s1, SeqPer)
    assert s1 == SeqPer((1, 4, 3, 2, 2, 6), (n, 1, 5))
    s2 = SeqMul(form, form_bou)
    assert isinstance(s2, SeqFormula)
    assert s2 == SeqFormula(n ** 4, (6, 10))
    assert SeqMul(form, form_bou, per) == SeqMul(per, SeqFormula(n ** 4, (6, 10)))
    assert SeqMul(form, SeqMul(form_bou, per)) == SeqMul(per, SeqFormula(n ** 4, (6, 10)))
    assert SeqMul(per, SeqMul(form, form_bou2, evaluate=False), evaluate=False) == SeqMul(form, per, form_bou2, evaluate=False)
    assert SeqMul(SeqPer((1, 2), (n, 0, oo)), SeqPer((1, 2), (n, 0, oo))) == SeqPer((1, 4), (n, 0, oo))