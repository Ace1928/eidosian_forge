import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_matcher_set_value_operator(en_vocab):
    matcher = Matcher(en_vocab)
    pattern = [{'ORTH': {'IN': ['a', 'the']}, 'OP': '?'}, {'ORTH': 'house'}]
    matcher.add('DET_HOUSE', [pattern])
    doc = Doc(en_vocab, words=['In', 'a', 'house'])
    matches = matcher(doc)
    assert len(matches) == 2
    doc = Doc(en_vocab, words=['my', 'house'])
    matches = matcher(doc)
    assert len(matches) == 1