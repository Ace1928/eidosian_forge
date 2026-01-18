import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_matcher_regex_shape(en_vocab):
    matcher = Matcher(en_vocab)
    pattern = [{'SHAPE': {'REGEX': '^[^x]+$'}}]
    matcher.add('NON_ALPHA', [pattern])
    doc = Doc(en_vocab, words=['99', 'problems', '!'])
    matches = matcher(doc)
    assert len(matches) == 2
    doc = Doc(en_vocab, words=['bye'])
    matches = matcher(doc)
    assert len(matches) == 0