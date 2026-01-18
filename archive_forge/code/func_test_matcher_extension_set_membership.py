import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_matcher_extension_set_membership(en_vocab):
    matcher = Matcher(en_vocab)
    get_reversed = lambda token: ''.join(reversed(token.text))
    Token.set_extension('reversed', getter=get_reversed, force=True)
    pattern = [{'_': {'reversed': {'IN': ['eyb', 'ih']}}}]
    matcher.add('REVERSED', [pattern])
    doc = Doc(en_vocab, words=['hi', 'bye', 'hello'])
    matches = matcher(doc)
    assert len(matches) == 2
    doc = Doc(en_vocab, words=['aardvark'])
    matches = matcher(doc)
    assert len(matches) == 0