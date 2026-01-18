import pytest
from spacy.errors import MatchPatternError
from spacy.matcher import Matcher
from spacy.schemas import validate_token_pattern
@pytest.mark.parametrize('pattern', [[{'XX': 'y'}], [{'LENGTH': '2'}], [{'TEXT': {'IN': 5}}], [{'text': {'in': 6}}]])
def test_matcher_pattern_validation(en_vocab, pattern):
    matcher = Matcher(en_vocab, validate=True)
    with pytest.raises(MatchPatternError):
        matcher.add('TEST', [pattern])