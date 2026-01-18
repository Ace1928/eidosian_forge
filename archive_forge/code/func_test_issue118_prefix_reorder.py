import re
import pytest
from spacy.attrs import IS_PUNCT, LOWER, ORTH
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.lang.lex_attrs import LEX_ATTRS
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
@pytest.mark.issue(118)
@pytest.mark.parametrize('patterns', [[[{'LOWER': 'boston'}], [{'LOWER': 'boston'}, {'LOWER': 'celtics'}]], [[{'LOWER': 'boston'}, {'LOWER': 'celtics'}], [{'LOWER': 'boston'}]]])
def test_issue118_prefix_reorder(en_tokenizer, patterns):
    """Test a bug that arose from having overlapping matches"""
    text = 'how many points did lebron james score against the boston celtics last night'
    doc = en_tokenizer(text)
    ORG = doc.vocab.strings['ORG']
    matcher = Matcher(doc.vocab)
    matcher.add('BostonCeltics', patterns)
    assert len(list(doc.ents)) == 0
    matches = [(ORG, start, end) for _, start, end in matcher(doc)]
    doc.ents += tuple(matches)[1:]
    assert matches == [(ORG, 9, 10), (ORG, 9, 11)]
    ents = doc.ents
    assert len(ents) == 1
    assert ents[0].label == ORG
    assert ents[0].start == 9
    assert ents[0].end == 11