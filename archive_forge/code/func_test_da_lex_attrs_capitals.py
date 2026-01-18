import pytest
from spacy.lang.da.lex_attrs import like_num
@pytest.mark.parametrize('word', ['elleve', 'første'])
def test_da_lex_attrs_capitals(word):
    assert like_num(word)
    assert like_num(word.upper())