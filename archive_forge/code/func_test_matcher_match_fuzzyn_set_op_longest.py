import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
@pytest.mark.parametrize('greedy', ['FIRST', 'LONGEST'])
@pytest.mark.parametrize('set_op', ['IN', 'NOT_IN'])
def test_matcher_match_fuzzyn_set_op_longest(en_vocab, greedy, set_op):
    rules = {'GoogleNow': [[{'ORTH': {'FUZZY2': {set_op: ['Google', 'Now']}}, 'OP': '+'}]]}
    matcher = Matcher(en_vocab)
    for key, patterns in rules.items():
        matcher.add(key, patterns, greedy=greedy)
    words = ['They', 'like', 'Goggle', 'Noo']
    doc = Doc(matcher.vocab, words=words)
    spans = matcher(doc, as_spans=True)
    assert len(spans) == 1
    if set_op == 'IN':
        assert spans[0].text == 'Goggle Noo'
    else:
        assert spans[0].text == 'They like'