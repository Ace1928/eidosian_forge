import pickle
import pytest
from netaddr import IPAddress, IPNetwork
def test_ipnetwork_constructor_v6():
    assert IPNetwork(IPNetwork('::192.0.2.0/120')) == IPNetwork('::192.0.2.0/120')
    assert IPNetwork('::192.0.2.0/120') == IPNetwork('::192.0.2.0/120')
    assert IPNetwork('::192.0.2.0/120', 6) == IPNetwork('::192.0.2.0/120')