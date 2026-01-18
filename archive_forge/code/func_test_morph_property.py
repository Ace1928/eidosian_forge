import pytest
def test_morph_property(tokenizer):
    doc = tokenizer('a dog')
    doc[0].set_morph('PronType=prs')
    assert str(doc[0].morph) == 'PronType=prs'
    assert doc.to_array(['MORPH'])[0] != 0
    doc[0].set_morph(None)
    assert doc.to_array(['MORPH'])[0] == 0
    doc[0].set_morph('')
    assert str(doc[0].morph) == ''
    assert doc.to_array(['MORPH'])[0] == tokenizer.vocab.strings['_']
    doc[0].set_morph('_')
    assert str(doc[0].morph) == ''
    assert doc.to_array(['MORPH'])[0] == tokenizer.vocab.strings['_']
    tokenizer.vocab.strings.add('Feat=Val')
    doc[0].set_morph(tokenizer.vocab.strings.add('Feat=Val'))
    assert str(doc[0].morph) == 'Feat=Val'