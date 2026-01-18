import pytest
from spacy.attrs import ENT_IOB, IS_ALPHA, LEMMA, NORM, ORTH, intify_attrs
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.lex_attrs import (
@pytest.mark.parametrize('word', ['the'])
@pytest.mark.issue(1889)
def test_issue1889(word):
    assert is_stop(word, STOP_WORDS) == is_stop(word.upper(), STOP_WORDS)