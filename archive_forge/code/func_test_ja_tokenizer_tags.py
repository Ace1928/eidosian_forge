import pytest
from spacy.lang.ja import DetailedToken, Japanese
from ...tokenizer.test_naughty_strings import NAUGHTY_STRINGS
@pytest.mark.parametrize('text,expected_tags', TAG_TESTS)
def test_ja_tokenizer_tags(ja_tokenizer, text, expected_tags):
    tags = [token.tag_ for token in ja_tokenizer(text)]
    assert tags == expected_tags