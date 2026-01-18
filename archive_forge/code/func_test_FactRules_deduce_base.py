from sympy.core.facts import (deduce_alpha_implications,
from sympy.core.logic import And, Not
from sympy.testing.pytest import raises
def test_FactRules_deduce_base():
    f = FactRules(['real  == neg | zero | pos', 'neg   -> real & !zero & !pos', 'pos   -> real & !zero & !neg'])
    base = FactKB(f)
    base.deduce_all_facts({'real': T, 'neg': F})
    assert base == {'real': T, 'neg': F}
    base.deduce_all_facts({'zero': F})
    assert base == {'real': T, 'neg': F, 'zero': F, 'pos': T}