import pytest
def test_morph_iter(i_has):
    assert set(i_has[0].morph) == set(['PronType=prs'])
    assert set(i_has[1].morph) == set(['Number=sing', 'Person=three', 'Tense=pres', 'VerbForm=fin'])