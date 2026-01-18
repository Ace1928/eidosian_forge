import pickle
import sys
import weakref
import pytest
from netaddr import IPAddress, IPNetwork, IPRange, IPSet, cidr_exclude
def test_ipset_is_weak_referencable():
    weakref.ref(IPSet())