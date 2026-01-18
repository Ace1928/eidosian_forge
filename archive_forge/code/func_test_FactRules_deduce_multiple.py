from sympy.core.facts import (deduce_alpha_implications,
from sympy.core.logic import And, Not
from sympy.testing.pytest import raises
def test_FactRules_deduce_multiple():
    f = FactRules(['real == pos | npos'])

    def D(facts):
        kb = FactKB(f)
        kb.deduce_all_facts(facts)
        return kb
    assert D({'real': T}) == {'real': T}
    assert D({'real': F}) == {'real': F, 'pos': F, 'npos': F}
    assert D({'pos': T}) == {'real': T, 'pos': T}
    assert D({'npos': T}) == {'real': T, 'npos': T}
    assert D({'pos': F, 'npos': F}) == {'real': F, 'pos': F, 'npos': F}
    assert D({'real': T, 'pos': F}) == {'real': T, 'pos': F, 'npos': T}
    assert D({'real': T, 'npos': F}) == {'real': T, 'pos': T, 'npos': F}
    assert D({'pos': T, 'npos': F}) == {'real': T, 'pos': T, 'npos': F}
    assert D({'pos': F, 'npos': T}) == {'real': T, 'pos': F, 'npos': T}