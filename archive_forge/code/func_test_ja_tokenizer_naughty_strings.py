import pytest
from spacy.lang.ja import DetailedToken, Japanese
from ...tokenizer.test_naughty_strings import NAUGHTY_STRINGS
@pytest.mark.parametrize('text', NAUGHTY_STRINGS)
def test_ja_tokenizer_naughty_strings(ja_tokenizer, text):
    tokens = ja_tokenizer(text)
    assert tokens.text_with_ws == text