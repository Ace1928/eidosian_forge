import pytest
from spacy.lang.ja import DetailedToken, Japanese
from ...tokenizer.test_naughty_strings import NAUGHTY_STRINGS
@pytest.mark.skip(reason='sentence segmentation in tokenizer is buggy')
@pytest.mark.parametrize('text,expected_sents', SENTENCE_TESTS)
def test_ja_tokenizer_sents(ja_tokenizer, text, expected_sents):
    sents = [str(sent) for sent in ja_tokenizer(text).sents]
    assert sents == expected_sents