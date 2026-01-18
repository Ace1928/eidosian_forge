import pickle
import sys
import weakref
import pytest
from netaddr import IPAddress, IPNetwork, IPRange, IPSet, cidr_exclude
def test_ipset_updates():
    s1 = IPSet(['192.0.2.0/25'])
    s2 = IPSet(['192.0.2.128/25'])
    s1.update(s2)
    assert s1 == IPSet(['192.0.2.0/24'])
    s1.update(['192.0.0.0/24', '192.0.1.0/24', '192.0.3.0/24'])
    assert s1 == IPSet(['192.0.0.0/22'])
    expected = IPSet(['192.0.1.0/24', '192.0.2.0/24'])
    s3 = IPSet(['192.0.1.0/24'])
    s3.update(IPRange('192.0.2.0', '192.0.2.255'))
    assert s3 == expected
    s4 = IPSet(['192.0.1.0/24'])
    s4.update([IPRange('192.0.2.0', '192.0.2.100'), IPRange('192.0.2.50', '192.0.2.255')])
    assert s4 == expected