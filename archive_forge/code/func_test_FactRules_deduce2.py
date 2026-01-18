from sympy.core.facts import (deduce_alpha_implications,
from sympy.core.logic import And, Not
from sympy.testing.pytest import raises
def test_FactRules_deduce2():
    f = FactRules(['pos -> !neg', 'pos -> !z'])

    def D(facts):
        kb = FactKB(f)
        kb.deduce_all_facts(facts)
        return kb
    assert D({'pos': T}) == {'pos': T, 'neg': F, 'z': F}
    assert D({'pos': F}) == {'pos': F}
    assert D({'neg': T}) == {'pos': F, 'neg': T}
    assert D({'neg': F}) == {'neg': F}
    assert D({'z': T}) == {'pos': F, 'z': T}
    assert D({'z': F}) == {'z': F}
    f = FactRules(['pos -> !neg', 'neg -> !pos', 'pos -> !z', 'neg -> !z'])
    assert D({'pos': T}) == {'pos': T, 'neg': F, 'z': F}
    assert D({'pos': F}) == {'pos': F}
    assert D({'neg': T}) == {'pos': F, 'neg': T, 'z': F}
    assert D({'neg': F}) == {'neg': F}
    assert D({'z': T}) == {'pos': F, 'neg': F, 'z': T}
    assert D({'z': F}) == {'z': F}