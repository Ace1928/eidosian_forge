import pytest
from spacy.lang.ja import DetailedToken, Japanese
from ...tokenizer.test_naughty_strings import NAUGHTY_STRINGS
def test_ja_tokenizer_emptyish_texts(ja_tokenizer):
    doc = ja_tokenizer('')
    assert len(doc) == 0
    doc = ja_tokenizer(' ')
    assert len(doc) == 1
    doc = ja_tokenizer('\n\n\n \t\t \n\n\n')
    assert len(doc) == 1