import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_matcher_extension_in_set_predicate(en_vocab):
    matcher = Matcher(en_vocab)
    Token.set_extension('ext', default=[])
    pattern = [{'_': {'ext': {'IN': ['A', 'C']}}}]
    matcher.add('M', [pattern])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    doc[0]._.ext = ['A', 'B']
    assert len(matcher(doc)) == 0
    doc[0]._.ext = ['A']
    assert len(matcher(doc)) == 0
    doc[0]._.ext = 'A'
    assert len(matcher(doc)) == 1