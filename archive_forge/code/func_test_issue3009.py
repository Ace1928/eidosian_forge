import re
import pytest
from spacy.attrs import IS_PUNCT, LOWER, ORTH
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.lang.lex_attrs import LEX_ATTRS
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
@pytest.mark.issue(3009)
def test_issue3009(en_vocab):
    """Test problem with matcher quantifiers"""
    patterns = [[{'ORTH': 'has'}, {'LOWER': 'to'}, {'LOWER': 'do'}, {'TAG': 'IN'}], [{'ORTH': 'has'}, {'IS_ASCII': True, 'IS_PUNCT': False, 'OP': '*'}, {'LOWER': 'to'}, {'LOWER': 'do'}, {'TAG': 'IN'}], [{'ORTH': 'has'}, {'IS_ASCII': True, 'IS_PUNCT': False, 'OP': '?'}, {'LOWER': 'to'}, {'LOWER': 'do'}, {'TAG': 'IN'}]]
    words = ['also', 'has', 'to', 'do', 'with']
    tags = ['RB', 'VBZ', 'TO', 'VB', 'IN']
    pos = ['ADV', 'VERB', 'ADP', 'VERB', 'ADP']
    doc = Doc(en_vocab, words=words, tags=tags, pos=pos)
    matcher = Matcher(en_vocab)
    for i, pattern in enumerate(patterns):
        matcher.add(str(i), [pattern])
        matches = matcher(doc)
        assert matches