from charset_normalizer.api import from_bytes
from charset_normalizer.models import CharsetMatches
import pytest
def test_mb_cutting_chk():
    payload = b'\xbf\xaa\xbb\xe7\xc0\xfb    \xbf\xb9\xbc\xf6    \xbf\xac\xb1\xb8\xc0\xda\xb5\xe9\xc0\xba  \xba\xb9\xc0\xbd\xbc\xad\xb3\xaa ' * 128
    guesses = from_bytes(payload, cp_isolation=['cp949'])
    best_guess = guesses.best()
    assert len(guesses) == 1, 'cp isolation is set and given seq should be clear CP949!'
    assert best_guess.encoding == 'cp949'