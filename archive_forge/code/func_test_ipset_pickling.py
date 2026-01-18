import pickle
import sys
import weakref
import pytest
from netaddr import IPAddress, IPNetwork, IPRange, IPSet, cidr_exclude
def test_ipset_pickling():
    ip_data = IPSet(['10.0.0.0/16', 'fe80::/64'])
    buf = pickle.dumps(ip_data)
    ip_data_unpickled = pickle.loads(buf)
    assert ip_data == ip_data_unpickled