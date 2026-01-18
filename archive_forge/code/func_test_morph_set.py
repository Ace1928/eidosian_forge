import pytest
def test_morph_set(i_has):
    assert i_has[0].morph.get('PronType') == ['prs']
    i_has[0].set_morph('PronType=unk')
    assert i_has[0].morph.get('PronType') == ['unk']
    i_has[0].set_morph('PronType=123|NounType=unk')
    assert str(i_has[0].morph) == 'NounType=unk|PronType=123'
    i_has[0].set_morph({'AType': '123', 'BType': 'unk'})
    assert str(i_has[0].morph) == 'AType=123|BType=unk'
    i_has[0].set_morph('BType=c|AType=b,a')
    assert str(i_has[0].morph) == 'AType=a,b|BType=c'
    i_has[0].set_morph({'AType': 'b,a', 'BType': 'c'})
    assert str(i_has[0].morph) == 'AType=a,b|BType=c'