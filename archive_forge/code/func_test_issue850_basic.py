import re
import pytest
from spacy.attrs import IS_PUNCT, LOWER, ORTH
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.lang.lex_attrs import LEX_ATTRS
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
@pytest.mark.issue(850)
def test_issue850_basic():
    """Test Matcher matches with '*' operator and Boolean flag"""
    vocab = Vocab(lex_attr_getters={LOWER: lambda string: string.lower()})
    matcher = Matcher(vocab)
    pattern = [{'LOWER': 'bob'}, {'OP': '*', 'LOWER': 'and'}, {'LOWER': 'frank'}]
    matcher.add('FarAway', [pattern])
    doc = Doc(matcher.vocab, words=['bob', 'and', 'and', 'frank'])
    match = matcher(doc)
    assert len(match) == 1
    ent_id, start, end = match[0]
    assert start == 0
    assert end == 4