import pickle
import types
import random
import pytest
from netaddr import (
def test_ipnetwork_equality_v4():
    assert IPNetwork('192.0.2.0/255.255.254.0') == IPNetwork('192.0.2.0/23')
    assert IPNetwork('192.0.2.65/255.255.254.0') == IPNetwork('192.0.2.0/23')
    assert IPNetwork('192.0.2.65/255.255.254.0') == IPNetwork('192.0.2.65/23')
    assert IPNetwork('192.0.2.65/255.255.255.0') != IPNetwork('192.0.2.0/23')
    assert IPNetwork('192.0.2.65/255.255.254.0') != IPNetwork('192.0.2.65/24')