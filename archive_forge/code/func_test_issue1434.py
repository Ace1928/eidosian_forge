import re
import pytest
from spacy.attrs import IS_PUNCT, LOWER, ORTH
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.lang.lex_attrs import LEX_ATTRS
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
@pytest.mark.issue(1434)
def test_issue1434():
    """Test matches occur when optional element at end of short doc."""
    pattern = [{'ORTH': 'Hello'}, {'IS_ALPHA': True, 'OP': '?'}]
    vocab = Vocab(lex_attr_getters=LEX_ATTRS)
    hello_world = Doc(vocab, words=['Hello', 'World'])
    hello = Doc(vocab, words=['Hello'])
    matcher = Matcher(vocab)
    matcher.add('MyMatcher', [pattern])
    matches = matcher(hello_world)
    assert matches
    matches = matcher(hello)
    assert matches