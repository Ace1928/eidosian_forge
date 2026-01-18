import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
@pytest.mark.parametrize('rules,match_locs', [({'GoogleNow': [[{'ORTH': {'FUZZY': 'Google'}}, {'ORTH': 'Now'}]]}, [(2, 4)]), ({'Java': [[{'LOWER': {'FUZZY': 'java'}}]]}, [(5, 6)]), ({'JS': [[{'ORTH': {'FUZZY': 'JavaScript'}}]], 'GoogleNow': [[{'ORTH': {'FUZZY': 'Google'}}, {'ORTH': 'Now'}]], 'Java': [[{'LOWER': {'FUZZY': 'java'}}]]}, [(2, 4), (5, 6), (8, 9)]), ({'A': [[{'ORTH': {'FUZZY': 'Javascripts'}}]], 'B': [[{'ORTH': {'FUZZY5': 'Javascripts'}}]]}, [(8, 9)])])
def test_matcher_match_fuzzy(en_vocab, rules, match_locs):
    words = ['They', 'like', 'Goggle', 'Now', 'and', 'Jav', 'but', 'not', 'JvvaScrpt']
    doc = Doc(en_vocab, words=words)
    matcher = Matcher(en_vocab)
    for key, patterns in rules.items():
        matcher.add(key, patterns)
    assert match_locs == [(start, end) for m_id, start, end in matcher(doc)]