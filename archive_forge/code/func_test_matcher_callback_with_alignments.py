import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_matcher_callback_with_alignments(en_vocab):
    mock = Mock()
    matcher = Matcher(en_vocab)
    pattern = [{'ORTH': 'test'}]
    matcher.add('Rule', [pattern], on_match=mock)
    doc = Doc(en_vocab, words=['This', 'is', 'a', 'test', '.'])
    matches = matcher(doc, with_alignments=True)
    mock.assert_called_once_with(matcher, doc, 0, matches)