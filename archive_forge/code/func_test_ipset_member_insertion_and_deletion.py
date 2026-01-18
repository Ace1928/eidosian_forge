import pickle
import sys
import weakref
import pytest
from netaddr import IPAddress, IPNetwork, IPRange, IPSet, cidr_exclude
def test_ipset_member_insertion_and_deletion():
    s1 = IPSet()
    s1.add('192.0.2.0')
    assert s1 == IPSet(['192.0.2.0/32'])
    s1.remove('192.0.2.0')
    assert s1 == IPSet([])
    s1.add(IPRange('10.0.0.0', '10.0.0.255'))
    assert s1 == IPSet(['10.0.0.0/24'])
    s1.remove(IPRange('10.0.0.128', '10.10.10.10'))
    assert s1 == IPSet(['10.0.0.0/25'])