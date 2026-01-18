import pickle
import sys
import weakref
import pytest
from netaddr import IPAddress, IPNetwork, IPRange, IPSet, cidr_exclude
def test_ipset_iteration():
    assert list(IPSet(['192.0.2.0/28', '::192.0.2.0/124'])) == [IPAddress('192.0.2.0'), IPAddress('192.0.2.1'), IPAddress('192.0.2.2'), IPAddress('192.0.2.3'), IPAddress('192.0.2.4'), IPAddress('192.0.2.5'), IPAddress('192.0.2.6'), IPAddress('192.0.2.7'), IPAddress('192.0.2.8'), IPAddress('192.0.2.9'), IPAddress('192.0.2.10'), IPAddress('192.0.2.11'), IPAddress('192.0.2.12'), IPAddress('192.0.2.13'), IPAddress('192.0.2.14'), IPAddress('192.0.2.15'), IPAddress('::192.0.2.0'), IPAddress('::192.0.2.1'), IPAddress('::192.0.2.2'), IPAddress('::192.0.2.3'), IPAddress('::192.0.2.4'), IPAddress('::192.0.2.5'), IPAddress('::192.0.2.6'), IPAddress('::192.0.2.7'), IPAddress('::192.0.2.8'), IPAddress('::192.0.2.9'), IPAddress('::192.0.2.10'), IPAddress('::192.0.2.11'), IPAddress('::192.0.2.12'), IPAddress('::192.0.2.13'), IPAddress('::192.0.2.14'), IPAddress('::192.0.2.15')]