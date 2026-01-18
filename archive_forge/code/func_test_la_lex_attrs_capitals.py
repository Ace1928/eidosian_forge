import pytest
from spacy.lang.la.lex_attrs import like_num
@pytest.mark.parametrize('word', ['quinque'])
def test_la_lex_attrs_capitals(word):
    assert like_num(word)
    assert like_num(word.upper())