import pickle
import types
import random
import pytest
from netaddr import (
def test_ipnetwork_index_operations_v4():
    ip = IPNetwork('192.0.2.16/29')
    assert ip[0] == IPAddress('192.0.2.16')
    assert ip[1] == IPAddress('192.0.2.17')
    assert ip[-1] == IPAddress('192.0.2.23')