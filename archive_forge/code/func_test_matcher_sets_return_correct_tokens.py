import re
import pytest
from spacy.attrs import IS_PUNCT, LOWER, ORTH
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.lang.lex_attrs import LEX_ATTRS
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
def test_matcher_sets_return_correct_tokens(en_vocab):
    matcher = Matcher(en_vocab)
    patterns = [[{'LOWER': {'IN': ['zero']}}], [{'LOWER': {'IN': ['one']}}], [{'LOWER': {'IN': ['two']}}]]
    matcher.add('TEST', patterns)
    doc = Doc(en_vocab, words='zero one two three'.split())
    matches = matcher(doc)
    texts = [Span(doc, s, e, label=L).text for L, s, e in matches]
    assert texts == ['zero', 'one', 'two']