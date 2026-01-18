import pickle
import sys
import weakref
import pytest
from netaddr import IPAddress, IPNetwork, IPRange, IPSet, cidr_exclude
def test_ipset_converts_to_cidr_networks_v4():
    s1 = IPSet(IPNetwork('10.1.2.3/8'))
    s1.add(IPNetwork('192.168.1.2/16'))
    assert list(s1.iter_cidrs()) == [IPNetwork('10.0.0.0/8'), IPNetwork('192.168.0.0/16')]