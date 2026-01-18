import pytest
from spacy.attrs import ENT_IOB, IS_ALPHA, LEMMA, NORM, ORTH, intify_attrs
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.lex_attrs import (
@pytest.mark.parametrize('text,match', [('$', True), ('£', True), ('♥', False), ('€', True), ('¥', True), ('¢', True), ('a', False), ('www.google.com', False), ('dog', False)])
def test_lex_attrs_is_currency(text, match):
    assert is_currency(text) == match