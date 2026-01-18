import pickle
import sys
import weakref
import pytest
from netaddr import IPAddress, IPNetwork, IPRange, IPSet, cidr_exclude
def test_ipset_constructor():
    assert IPSet(['192.0.2.0']) == IPSet(['192.0.2.0/32'])
    assert IPSet([IPAddress('192.0.2.0')]) == IPSet(['192.0.2.0/32'])
    assert IPSet([IPNetwork('192.0.2.0')]) == IPSet(['192.0.2.0/32'])
    assert IPSet(IPNetwork('1234::/32')) == IPSet(['1234::/32'])
    assert IPSet([IPNetwork('192.0.2.0/24')]) == IPSet(['192.0.2.0/24'])
    assert IPSet(IPSet(['192.0.2.0/32'])) == IPSet(['192.0.2.0/32'])
    assert IPSet(IPRange('10.0.0.0', '10.0.1.31')) == IPSet(['10.0.0.0/24', '10.0.1.0/27'])
    assert IPSet(IPRange('0.0.0.0', '255.255.255.255')) == IPSet(['0.0.0.0/0'])
    assert IPSet([IPRange('10.0.0.0', '10.0.1.31')]) == IPSet(IPRange('10.0.0.0', '10.0.1.31'))