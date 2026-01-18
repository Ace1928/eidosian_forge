import pytest
def test_morph_str(i_has):
    assert str(i_has[0].morph) == 'PronType=prs'
    assert str(i_has[1].morph) == 'Number=sing|Person=three|Tense=pres|VerbForm=fin'