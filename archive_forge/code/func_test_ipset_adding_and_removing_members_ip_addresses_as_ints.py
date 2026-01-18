import pickle
import sys
import weakref
import pytest
from netaddr import IPAddress, IPNetwork, IPRange, IPSet, cidr_exclude
def test_ipset_adding_and_removing_members_ip_addresses_as_ints():
    s1 = IPSet(['10.0.0.0/25'])
    s1.add('10.0.0.0/24')
    assert s1 == IPSet(['10.0.0.0/24'])
    integer1 = int(IPAddress('10.0.0.1'))
    integer2 = int(IPAddress('fe80::'))
    integer3 = int(IPAddress('10.0.0.2'))
    s2 = IPSet([integer1, integer2])
    assert s2 == IPSet(['10.0.0.1/32', 'fe80::/128'])
    s2.add(integer3)
    assert s2 == IPSet(['10.0.0.1/32', '10.0.0.2/32', 'fe80::/128'])
    s2.remove(integer2)
    assert s2 == IPSet(['10.0.0.1/32', '10.0.0.2/32'])
    s2.update([integer2])
    assert s2 == IPSet(['10.0.0.1/32', '10.0.0.2/32', 'fe80::/128'])