import pytest
from spacy.lang.en.lex_attrs import like_num
@pytest.mark.parametrize('word', ['eleven'])
def test_en_lex_attrs_capitals(word):
    assert like_num(word)
    assert like_num(word.upper())