import re
import pytest
from spacy.attrs import IS_PUNCT, LOWER, ORTH
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.lang.lex_attrs import LEX_ATTRS
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
@pytest.mark.skip('Matching currently only works on strings and integers')
@pytest.mark.issue(3555)
def test_issue3555(en_vocab):
    """Test that custom extensions with default None don't break matcher."""
    Token.set_extension('issue3555', default=None)
    matcher = Matcher(en_vocab)
    pattern = [{'ORTH': 'have'}, {'_': {'issue3555': True}}]
    matcher.add('TEST', [pattern])
    doc = Doc(en_vocab, words=['have', 'apple'])
    matcher(doc)