import pickle
import sys
import weakref
import pytest
from netaddr import IPAddress, IPNetwork, IPRange, IPSet, cidr_exclude
def test_ipset_converts_to_cidr_networks_v6():
    s1 = IPSet(IPNetwork('fe80::4242/64'))
    s1.add(IPNetwork('fe90::4343/64'))
    assert list(s1.iter_cidrs()) == [IPNetwork('fe80::/64'), IPNetwork('fe90::/64')]