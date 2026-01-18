import pickle
import types
import random
import pytest
from netaddr import (
def test_ipnetwork_sort_order():
    ip_list = list(IPNetwork('192.0.2.128/28'))
    random.shuffle(ip_list)
    assert sorted(ip_list) == [IPAddress('192.0.2.128'), IPAddress('192.0.2.129'), IPAddress('192.0.2.130'), IPAddress('192.0.2.131'), IPAddress('192.0.2.132'), IPAddress('192.0.2.133'), IPAddress('192.0.2.134'), IPAddress('192.0.2.135'), IPAddress('192.0.2.136'), IPAddress('192.0.2.137'), IPAddress('192.0.2.138'), IPAddress('192.0.2.139'), IPAddress('192.0.2.140'), IPAddress('192.0.2.141'), IPAddress('192.0.2.142'), IPAddress('192.0.2.143')]