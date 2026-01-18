import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_matcher_any_token_operator(en_vocab):
    """Test that patterns with "any token" {} work with operators."""
    matcher = Matcher(en_vocab)
    matcher.add('TEST', [[{'ORTH': 'test'}, {'OP': '*'}]])
    doc = Doc(en_vocab, words=['test', 'hello', 'world'])
    matches = [doc[start:end].text for _, start, end in matcher(doc)]
    assert len(matches) == 3
    assert matches[0] == 'test'
    assert matches[1] == 'test hello'
    assert matches[2] == 'test hello world'