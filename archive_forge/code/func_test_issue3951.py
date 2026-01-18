import re
import pytest
from spacy.attrs import IS_PUNCT, LOWER, ORTH
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.lang.lex_attrs import LEX_ATTRS
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
@pytest.mark.issue(3951)
def test_issue3951(en_vocab):
    """Test that combinations of optional rules are matched correctly."""
    matcher = Matcher(en_vocab)
    pattern = [{'LOWER': 'hello'}, {'LOWER': 'this', 'OP': '?'}, {'OP': '?'}, {'LOWER': 'world'}]
    matcher.add('TEST', [pattern])
    doc = Doc(en_vocab, words=['Hello', 'my', 'new', 'world'])
    matches = matcher(doc)
    assert len(matches) == 0