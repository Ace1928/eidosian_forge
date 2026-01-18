import pickle
import sys
import weakref
import pytest
from netaddr import IPAddress, IPNetwork, IPRange, IPSet, cidr_exclude
def test_ipset_clear():
    ipset = IPSet(['10.0.0.0/16'])
    ipset.update(IPRange('10.1.0.0', '10.1.255.255'))
    assert ipset == IPSet(['10.0.0.0/15'])
    ipset.clear()
    assert ipset == IPSet([])