import pickle
import types
import random
import pytest
from netaddr import (
def test_ipnetwork_pickling_v4():
    cidr = IPNetwork('192.0.2.0/24')
    assert cidr == IPNetwork('192.0.2.0/24')
    buf = pickle.dumps(cidr)
    cidr2 = pickle.loads(buf)
    assert cidr2 == cidr
    assert id(cidr2) != id(cidr)
    assert cidr2.value == 3221225984
    assert cidr2.prefixlen == 24
    assert cidr2.version == 4