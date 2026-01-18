import pickle
import sys
import weakref
import pytest
from netaddr import IPAddress, IPNetwork, IPRange, IPSet, cidr_exclude
def test_ipset_ipv4_and_ipv4_separation():
    assert list(IPSet([IPAddress(1, 4), IPAddress(1, 6)]).iter_ipranges()) == [IPRange('0.0.0.1', '0.0.0.1'), IPRange('::1', '::1')]