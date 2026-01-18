import pickle
import sys
import weakref
import pytest
from netaddr import IPAddress, IPNetwork, IPRange, IPSet, cidr_exclude
def test_converting_ipsets_to_ipranges():
    assert list(IPSet().iter_ipranges()) == []
    assert list(IPSet([IPAddress('10.0.0.1')]).iter_ipranges()) == [IPRange('10.0.0.1', '10.0.0.1')]
    assert list(IPSet([IPAddress('10.0.0.1'), IPAddress('10.0.0.2')]).iter_ipranges()) == [IPRange('10.0.0.1', '10.0.0.2')]