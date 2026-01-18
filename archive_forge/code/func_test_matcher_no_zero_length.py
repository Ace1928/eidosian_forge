import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_matcher_no_zero_length(en_vocab):
    doc = Doc(en_vocab, words=['a', 'b'], tags=['A', 'B'])
    matcher = Matcher(en_vocab)
    matcher.add('TEST', [[{'TAG': 'C', 'OP': '?'}]])
    assert len(matcher(doc)) == 0