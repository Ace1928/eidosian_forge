import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_matcher_no_match(matcher):
    doc = Doc(matcher.vocab, words=['I', 'like', 'cheese', '.'])
    assert matcher(doc) == []