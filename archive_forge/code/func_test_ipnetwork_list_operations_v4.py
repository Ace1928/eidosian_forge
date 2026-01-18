import pickle
import types
import random
import pytest
from netaddr import (
def test_ipnetwork_list_operations_v4():
    ip = IPNetwork('192.0.2.16/29')
    assert len(ip) == 8
    ip_list = list(ip)
    assert len(ip_list) == 8
    assert ip_list == [IPAddress('192.0.2.16'), IPAddress('192.0.2.17'), IPAddress('192.0.2.18'), IPAddress('192.0.2.19'), IPAddress('192.0.2.20'), IPAddress('192.0.2.21'), IPAddress('192.0.2.22'), IPAddress('192.0.2.23')]