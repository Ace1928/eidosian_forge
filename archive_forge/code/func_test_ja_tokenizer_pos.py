import pytest
from spacy.lang.ja import DetailedToken, Japanese
from ...tokenizer.test_naughty_strings import NAUGHTY_STRINGS
@pytest.mark.parametrize('text,expected_pos', POS_TESTS)
def test_ja_tokenizer_pos(ja_tokenizer, text, expected_pos):
    pos = [token.pos_ for token in ja_tokenizer(text)]
    assert pos == expected_pos