import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_matcher_empty_dict(en_vocab):
    """Test matcher allows empty token specs, meaning match on any token."""
    matcher = Matcher(en_vocab)
    doc = Doc(matcher.vocab, words=['a', 'b', 'c'])
    matcher.add('A.C', [[{'ORTH': 'a'}, {}, {'ORTH': 'c'}]])
    matches = matcher(doc)
    assert len(matches) == 1
    assert matches[0][1:] == (0, 3)
    matcher = Matcher(en_vocab)
    matcher.add('A.', [[{'ORTH': 'a'}, {}]])
    matches = matcher(doc)
    assert matches[0][1:] == (0, 2)