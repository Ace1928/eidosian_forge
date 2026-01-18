import re
import pytest
from spacy.attrs import IS_PUNCT, LOWER, ORTH
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.lang.lex_attrs import LEX_ATTRS
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
@pytest.mark.issue(1945)
def test_issue1945():
    """Test regression in Matcher introduced in v2.0.6."""
    matcher = Matcher(Vocab())
    matcher.add('MWE', [[{'orth': 'a'}, {'orth': 'a'}]])
    doc = Doc(matcher.vocab, words=['a', 'a', 'a'])
    matches = matcher(doc)
    assert len(matches) == 2
    assert matches[0][1:] == (0, 2)
    assert matches[1][1:] == (1, 3)