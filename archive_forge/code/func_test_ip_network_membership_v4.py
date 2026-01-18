import pickle
import types
import random
import pytest
from netaddr import (
def test_ip_network_membership_v4():
    for what, network, result in [(IPAddress('192.0.2.1'), IPNetwork('192.0.2.0/24'), True), (IPAddress('192.0.2.255'), IPNetwork('192.0.2.0/24'), True), (IPNetwork('192.0.2.0/24'), IPNetwork('192.0.2.0/23'), True), (IPNetwork('192.0.2.0/24'), IPNetwork('192.0.2.0/24'), True), (IPNetwork('192.0.2.0/23'), IPNetwork('192.0.2.0/24'), False)]:
        assert (what in network) is result
        assert (str(what) in network) is result