import pickle
import types
import random
import pytest
from netaddr import (
def test_ipnetwork_change_netmask():
    ip = IPNetwork('192.168.0.0/16')
    ip.netmask = '255.0.0.0'
    assert ip.prefixlen == 8
    ip = IPNetwork('dead:beef::/16')
    ip.netmask = 'ffff:ffff:ffff:ffff::'
    assert ip.prefixlen == 64