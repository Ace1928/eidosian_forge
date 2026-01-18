from ast import literal_eval
import pickle
import sys
import sys
import pytest
from netaddr import (
def test_iprange_invalid_len_and_alternative():
    range1 = IPRange(IPAddress('::0'), IPAddress(sys.maxsize, 6))
    with pytest.raises(IndexError):
        len(range1)
    range2 = IPRange(IPAddress('::0'), IPAddress(sys.maxsize - 1, 6))
    assert len(range2) == sys.maxsize