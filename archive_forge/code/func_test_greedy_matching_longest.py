import re
import pytest
from spacy.attrs import IS_PUNCT, LOWER, ORTH
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.lang.lex_attrs import LEX_ATTRS
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
@pytest.mark.parametrize('pattern,longest', [(pattern1, longest1), (pattern2, longest2), (pattern3, longest3), (pattern4, longest4), (pattern5, longest5)])
def test_greedy_matching_longest(doc, text, pattern, longest):
    """Test the "LONGEST" greedy matching behavior"""
    matcher = Matcher(doc.vocab)
    matcher.add('RULE', [pattern], greedy='LONGEST')
    matches = matcher(doc)
    for key, s, e in matches:
        assert doc[s:e].text == longest