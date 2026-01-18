import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_matcher_match_one_plus(matcher):
    control = Matcher(matcher.vocab)
    control.add('BasicPhilippe', [[{'ORTH': 'Philippe'}]])
    doc = Doc(control.vocab, words=['Philippe', 'Philippe'])
    m = control(doc)
    assert len(m) == 2
    pattern = [{'ORTH': 'Philippe'}, {'ORTH': 'Philippe', 'OP': '+'}]
    matcher.add('KleenePhilippe', [pattern])
    m = matcher(doc)
    assert len(m) == 1