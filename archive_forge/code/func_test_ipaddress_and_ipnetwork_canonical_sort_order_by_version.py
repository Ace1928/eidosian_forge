import pickle
import types
import random
import pytest
from netaddr import (
def test_ipaddress_and_ipnetwork_canonical_sort_order_by_version():
    ip_list = [IPAddress('192.0.2.130'), IPNetwork('192.0.2.128/28'), IPAddress('::'), IPNetwork('192.0.3.0/24'), IPNetwork('192.0.2.0/24'), IPNetwork('fe80::/64'), IPAddress('10.0.0.1')]
    random.shuffle(ip_list)
    ip_list.sort()
    assert ip_list == [IPAddress('10.0.0.1'), IPNetwork('192.0.2.0/24'), IPNetwork('192.0.2.128/28'), IPAddress('192.0.2.130'), IPNetwork('192.0.3.0/24'), IPAddress('::'), IPNetwork('fe80::/64')]