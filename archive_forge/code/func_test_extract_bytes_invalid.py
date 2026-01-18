import pytest
from cheroot._compat import extract_bytes, ntob, ntou, bton
def test_extract_bytes_invalid():
    """Ensure that invalid input causes exception to be raised."""
    with pytest.raises(ValueError, match='^extract_bytes\\(\\) only accepts bytes and memoryview/buffer$'):
        extract_bytes(u'some юнікод їїї')