import pickle
import types
import random
import pytest
from netaddr import (
def test_ipnetwork_constructor_v4():
    assert IPNetwork('192.0.2.0/24') == IPNetwork('192.0.2.0/24')
    assert IPNetwork('192.0.2.0/255.255.255.0') == IPNetwork('192.0.2.0/24')
    assert IPNetwork('192.0.2.0/0.0.0.255') == IPNetwork('192.0.2.0/24')
    assert IPNetwork(IPNetwork('192.0.2.0/24')) == IPNetwork('192.0.2.0/24')
    assert IPNetwork(IPNetwork('192.0.2.0/24')) == IPNetwork('192.0.2.0/24')