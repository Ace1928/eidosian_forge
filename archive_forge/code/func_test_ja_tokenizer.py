import pytest
from spacy.lang.ja import DetailedToken, Japanese
from ...tokenizer.test_naughty_strings import NAUGHTY_STRINGS
@pytest.mark.parametrize('text,expected_tokens', TOKENIZER_TESTS)
def test_ja_tokenizer(ja_tokenizer, text, expected_tokens):
    tokens = [token.text for token in ja_tokenizer(text)]
    assert tokens == expected_tokens