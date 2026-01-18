import pickle
import types
import random
import pytest
from netaddr import (
def test_ipnetwork_change_prefixlen():
    ip = IPNetwork('192.168.0.0/16')
    assert ip.prefixlen == 16
    ip.prefixlen = 8
    assert ip.prefixlen == 8
    ip = IPNetwork('dead:beef::/16')
    assert ip.prefixlen == 16
    ip.prefixlen = 64
    assert ip.prefixlen == 64