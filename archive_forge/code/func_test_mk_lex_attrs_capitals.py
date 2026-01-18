import pytest
from spacy.lang.mk.lex_attrs import like_num
@pytest.mark.parametrize('word', ['двесте', 'два-три', 'пет-шест'])
def test_mk_lex_attrs_capitals(word):
    assert like_num(word)
    assert like_num(word.upper())