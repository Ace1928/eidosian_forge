import re
import pytest
from spacy.attrs import IS_PUNCT, LOWER, ORTH
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.lang.lex_attrs import LEX_ATTRS
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
@pytest.mark.issue(587)
def test_issue587(en_tokenizer):
    """Test that Matcher doesn't segfault on particular input"""
    doc = en_tokenizer('a b; c')
    matcher = Matcher(doc.vocab)
    matcher.add('TEST1', [[{ORTH: 'a'}, {ORTH: 'b'}]])
    matches = matcher(doc)
    assert len(matches) == 1
    matcher.add('TEST2', [[{ORTH: 'a'}, {ORTH: 'b'}, {IS_PUNCT: True}, {ORTH: 'c'}]])
    matches = matcher(doc)
    assert len(matches) == 2
    matcher.add('TEST3', [[{ORTH: 'a'}, {ORTH: 'b'}, {IS_PUNCT: True}, {ORTH: 'd'}]])
    matches = matcher(doc)
    assert len(matches) == 2