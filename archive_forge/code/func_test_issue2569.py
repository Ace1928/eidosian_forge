import re
import pytest
from spacy.attrs import IS_PUNCT, LOWER, ORTH
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.lang.lex_attrs import LEX_ATTRS
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
@pytest.mark.issue(2569)
def test_issue2569(en_tokenizer):
    """Test that operator + is greedy."""
    doc = en_tokenizer('It is May 15, 1993.')
    doc.ents = [Span(doc, 2, 6, label=doc.vocab.strings['DATE'])]
    matcher = Matcher(doc.vocab)
    matcher.add('RULE', [[{'ENT_TYPE': 'DATE', 'OP': '+'}]])
    matched = [doc[start:end] for _, start, end in matcher(doc)]
    matched = sorted(matched, key=len, reverse=True)
    assert len(matched) == 10
    assert len(matched[0]) == 4
    assert matched[0].text == 'May 15, 1993'