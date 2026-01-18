import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_matcher_regex_set_not_in(en_vocab):
    matcher = Matcher(en_vocab)
    pattern = [{'ORTH': {'REGEX': {'NOT_IN': ['(?:a)', '(?:an)']}}}]
    matcher.add('A_OR_AN', [pattern])
    doc = Doc(en_vocab, words=['an', 'a', 'hi'])
    matches = matcher(doc)
    assert len(matches) == 1
    doc = Doc(en_vocab, words=['bye'])
    matches = matcher(doc)
    assert len(matches) == 1