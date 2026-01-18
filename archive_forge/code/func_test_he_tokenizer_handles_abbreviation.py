import pytest
from spacy.lang.he.lex_attrs import like_num
@pytest.mark.parametrize('text,expected_tokens', [('פייתון היא שפת תכנות דינמית', ['פייתון', 'היא', 'שפת', 'תכנות', 'דינמית'])])
def test_he_tokenizer_handles_abbreviation(he_tokenizer, text, expected_tokens):
    tokens = he_tokenizer(text)
    token_list = [token.text for token in tokens if not token.is_space]
    assert expected_tokens == token_list