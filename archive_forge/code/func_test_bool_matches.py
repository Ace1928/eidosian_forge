from charset_normalizer.api import from_bytes
from charset_normalizer.models import CharsetMatches
import pytest
def test_bool_matches():
    guesses_not_empty = from_bytes(b'')
    guesses_empty = CharsetMatches([])
    assert bool(guesses_not_empty) is True, 'Bool behaviour of CharsetMatches altered, should be True'
    assert bool(guesses_empty) is False, 'Bool behaviour of CharsetMatches altered, should be False'