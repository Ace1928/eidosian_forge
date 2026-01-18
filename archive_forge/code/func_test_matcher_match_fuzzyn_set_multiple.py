import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_matcher_match_fuzzyn_set_multiple(en_vocab):
    rules = {'GoogleNow': [[{'ORTH': {'FUZZY1': {'IN': ['Google', 'Now']}, 'NOT_IN': ['Goggle']}, 'OP': '+'}]]}
    matcher = Matcher(en_vocab)
    for key, patterns in rules.items():
        matcher.add(key, patterns, greedy='LONGEST')
    words = ['They', 'like', 'Goggle', 'Noo']
    doc = Doc(matcher.vocab, words=words)
    assert matcher(doc) == [(doc.vocab.strings['GoogleNow'], 3, 4)]