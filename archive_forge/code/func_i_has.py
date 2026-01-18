import pytest
@pytest.fixture
def i_has(en_tokenizer):
    doc = en_tokenizer('I has')
    doc[0].set_morph({'PronType': 'prs'})
    doc[1].set_morph({'VerbForm': 'fin', 'Tense': 'pres', 'Number': 'sing', 'Person': 'three'})
    return doc