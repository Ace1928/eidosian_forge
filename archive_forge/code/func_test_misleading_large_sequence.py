import pytest
from charset_normalizer import from_bytes
from charset_normalizer.constant import TOO_BIG_SEQUENCE
def test_misleading_large_sequence():
    content = ('hello simple ascii ' * TOO_BIG_SEQUENCE + '我没有埋怨，磋砣的只是一些时间。 磋砣的只是一些时间。').encode('utf_8')
    guesses = from_bytes(content)
    assert len(guesses) > 0
    match = guesses.best()
    assert match is not None
    assert match.encoding == 'utf_8'
    assert str(match) is not None