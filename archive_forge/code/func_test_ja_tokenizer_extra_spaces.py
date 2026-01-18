import pytest
from spacy.lang.ja import DetailedToken, Japanese
from ...tokenizer.test_naughty_strings import NAUGHTY_STRINGS
def test_ja_tokenizer_extra_spaces(ja_tokenizer):
    tokens = ja_tokenizer('I   like cheese.')
    assert tokens[1].orth_ == '  '