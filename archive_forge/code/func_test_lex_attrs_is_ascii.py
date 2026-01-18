import pytest
from spacy.attrs import ENT_IOB, IS_ALPHA, LEMMA, NORM, ORTH, intify_attrs
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.lex_attrs import (
@pytest.mark.parametrize('text,match', [(',', True), ('£', False), ('♥', False)])
def test_lex_attrs_is_ascii(text, match):
    assert is_ascii(text) == match