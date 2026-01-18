import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_matcher_span(matcher):
    text = 'JavaScript is good but Java is better'
    doc = Doc(matcher.vocab, words=text.split())
    span_js = doc[:3]
    span_java = doc[4:]
    assert len(matcher(doc)) == 2
    assert len(matcher(span_js)) == 1
    assert len(matcher(span_java)) == 1