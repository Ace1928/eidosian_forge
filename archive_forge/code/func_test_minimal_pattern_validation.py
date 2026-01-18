import pytest
from spacy.errors import MatchPatternError
from spacy.matcher import Matcher
from spacy.schemas import validate_token_pattern
@pytest.mark.parametrize('pattern,n_errors,n_min_errors', TEST_PATTERNS)
def test_minimal_pattern_validation(en_vocab, pattern, n_errors, n_min_errors):
    matcher = Matcher(en_vocab)
    if n_min_errors > 0:
        with pytest.raises(ValueError):
            matcher.add('TEST', [pattern])
    elif n_errors == 0:
        matcher.add('TEST', [pattern])