import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_matcher_empty_patterns_warns(en_vocab):
    matcher = Matcher(en_vocab)
    assert len(matcher) == 0
    doc = Doc(en_vocab, words=['This', 'is', 'quite', 'something'])
    with pytest.warns(UserWarning):
        matcher(doc)
    assert len(doc.ents) == 0