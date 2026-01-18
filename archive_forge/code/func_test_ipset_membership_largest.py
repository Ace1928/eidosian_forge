import pickle
import sys
import weakref
import pytest
from netaddr import IPAddress, IPNetwork, IPRange, IPSet, cidr_exclude
def test_ipset_membership_largest():
    ipset = IPSet(['0.0.0.0/0'])
    assert IPAddress('10.0.0.1') in ipset
    assert IPAddress('0.0.0.0') in ipset
    assert IPAddress('255.255.255.0') in ipset
    assert IPNetwork('10.0.0.0/24') in ipset
    assert IPAddress('::1') not in ipset