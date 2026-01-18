import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_matcher_remove_zero_operator(en_vocab):
    matcher = Matcher(en_vocab)
    pattern = [{'OP': '!'}]
    matcher.add('Rule', [pattern])
    doc = Doc(en_vocab, words=['This', 'is', 'a', 'test', '.'])
    matches = matcher(doc)
    assert len(matches) == 0
    assert 'Rule' in matcher
    matcher.remove('Rule')
    assert 'Rule' not in matcher