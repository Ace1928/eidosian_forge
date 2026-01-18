import re
import pytest
from spacy.attrs import IS_PUNCT, LOWER, ORTH
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.lang.lex_attrs import LEX_ATTRS
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
@pytest.mark.parametrize('string,start,end', [('a', 0, 1), ('a b', 0, 2), ('a c', 0, 1), ('a b c', 0, 2), ('a b b c', 0, 3), ('a b b', 0, 3)])
@pytest.mark.issue(1450)
def test_issue1450(string, start, end):
    """Test matcher works when patterns end with * operator."""
    pattern = [{'ORTH': 'a'}, {'ORTH': 'b', 'OP': '*'}]
    matcher = Matcher(Vocab())
    matcher.add('TSTEND', [pattern])
    doc = Doc(Vocab(), words=string.split())
    matches = matcher(doc)
    if start is None or end is None:
        assert matches == []
    assert matches[-1][1] == start
    assert matches[-1][2] == end