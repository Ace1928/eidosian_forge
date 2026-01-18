import pytest
def test_morph_props(i_has):
    assert i_has[0].morph.get('PronType') == ['prs']
    assert i_has[1].morph.get('PronType') == []
    assert i_has[1].morph.get('AsdfType', ['asdf']) == ['asdf']
    assert i_has[1].morph.get('AsdfType', default=['asdf', 'qwer']) == ['asdf', 'qwer']