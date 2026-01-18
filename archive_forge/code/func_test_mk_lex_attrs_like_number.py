import pytest
from spacy.lang.mk.lex_attrs import like_num
@pytest.mark.parametrize('word,match', [('10', True), ('1', True), ('10.000', True), ('1000', True), ('бројка', False), ('999,0', True), ('еден', True), ('два', True), ('цифра', False), ('десет', True), ('сто', True), ('број', False), ('илјада', True), ('илјади', True), ('милион', True), (',', False), ('милијарда', True), ('билион', True)])
def test_mk_lex_attrs_like_number(mk_tokenizer, word, match):
    tokens = mk_tokenizer(word)
    assert len(tokens) == 1
    assert tokens[0].like_num == match