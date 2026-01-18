import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
@pytest.mark.parametrize('fuzzyn', range(1, 10))
def test_matcher_match_fuzzyn_all_insertions(en_vocab, fuzzyn):
    matcher = Matcher(en_vocab)
    matcher.add('GoogleNow', [[{'ORTH': {f'FUZZY{fuzzyn}': 'GoogleNow'}}]])
    words = ['GoogleNow' + 'a' * i for i in range(0, 10)]
    doc = Doc(en_vocab, words)
    assert len(matcher(doc)) == fuzzyn + 1