import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
@pytest.mark.parametrize('set_op', ['IN', 'NOT_IN'])
def test_matcher_match_fuzzy_set_op_longest(en_vocab, set_op):
    rules = {'GoogleNow': [[{'ORTH': {'FUZZY': {set_op: ['Google', 'Now']}}, 'OP': '+'}]]}
    matcher = Matcher(en_vocab)
    for key, patterns in rules.items():
        matcher.add(key, patterns, greedy='LONGEST')
    words = ['They', 'like', 'Goggle', 'Noo']
    doc = Doc(en_vocab, words=words)
    assert len(matcher(doc)) == 1