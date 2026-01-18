from ast import literal_eval
import pickle
import sys
import sys
import pytest
from netaddr import (
def test_iprange_indexing():
    iprange = IPRange('192.0.2.1', '192.0.2.254')
    assert len(iprange) == 254
    assert iprange.first == 3221225985
    assert iprange.last == 3221226238
    assert iprange[0] == IPAddress('192.0.2.1')
    assert iprange[-1] == IPAddress('192.0.2.254')
    with pytest.raises(IndexError):
        iprange[512]