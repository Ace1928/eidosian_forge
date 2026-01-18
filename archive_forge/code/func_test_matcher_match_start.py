import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_matcher_match_start(matcher):
    doc = Doc(matcher.vocab, words=['JavaScript', 'is', 'good'])
    assert matcher(doc) == [(matcher.vocab.strings['JS'], 0, 1)]