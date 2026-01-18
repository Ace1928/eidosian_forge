import re
import pytest
from spacy.attrs import IS_PUNCT, LOWER, ORTH
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.lang.lex_attrs import LEX_ATTRS
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
@pytest.mark.issue(3879)
def test_issue3879(en_vocab):
    doc = Doc(en_vocab, words=['This', 'is', 'a', 'test', '.'])
    assert len(doc) == 5
    pattern = [{'ORTH': 'This', 'OP': '?'}, {'OP': '?'}, {'ORTH': 'test'}]
    matcher = Matcher(en_vocab)
    matcher.add('TEST', [pattern])
    assert len(matcher(doc)) == 2