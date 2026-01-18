import re
import pytest
from spacy.attrs import IS_PUNCT, LOWER, ORTH
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.lang.lex_attrs import LEX_ATTRS
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
def test_greedy_matching_longest_first(en_tokenizer):
    """Test that "LONGEST" matching prefers the first of two equally long matches"""
    doc = en_tokenizer(' '.join('CCC'))
    matcher = Matcher(doc.vocab)
    pattern = [{'ORTH': 'C'}, {'ORTH': 'C'}]
    matcher.add('RULE', [pattern], greedy='LONGEST')
    matches = matcher(doc)
    assert len(matches) == 1
    assert matches[0][1] == 0
    assert matches[0][2] == 2