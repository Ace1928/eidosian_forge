import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_matcher_len_contains(matcher):
    assert len(matcher) == 3
    matcher.add('TEST', [[{'ORTH': 'test'}]])
    assert 'TEST' in matcher
    assert 'TEST2' not in matcher