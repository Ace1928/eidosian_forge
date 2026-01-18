from sympy.core.facts import (deduce_alpha_implications,
from sympy.core.logic import And, Not
from sympy.testing.pytest import raises
def test_FactRules_deduce_staticext():
    f = FactRules(['real  == neg | zero | pos', 'neg   -> real & !zero & !pos', 'pos   -> real & !zero & !neg', 'nneg  == real & !neg', 'npos  == real & !pos'])
    assert ('npos', True) in f.full_implications['neg', True]
    assert ('nneg', True) in f.full_implications['pos', True]
    assert ('nneg', True) in f.full_implications['zero', True]
    assert ('npos', True) in f.full_implications['zero', True]