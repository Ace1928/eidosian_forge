import pytest
from spacy.attrs import ENT_IOB, IS_ALPHA, LEMMA, NORM, ORTH, intify_attrs
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.lex_attrs import (
@pytest.mark.parametrize('text', ['dog'])
def test_attrs_key(text):
    assert intify_attrs({'ORTH': text}) == {ORTH: text}
    assert intify_attrs({'NORM': text}) == {NORM: text}
    assert intify_attrs({'lemma': text}, strings_map={text: 10}) == {LEMMA: 10}