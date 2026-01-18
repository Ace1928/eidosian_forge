import pytest
from spacy.lang.es import Spanish
from spacy.lang.es.lex_attrs import like_num
@pytest.mark.parametrize('word', ['once'])
def test_es_lex_attrs_capitals(word):
    assert like_num(word)
    assert like_num(word.upper())