import pickle
import types
import random
import pytest
from netaddr import (
def test_ipnetwork_incrementing_by_int():
    ip = IPNetwork('192.0.2.0/28')
    results = []
    for i in range(16):
        results.append(str(ip))
        ip += 1
    assert results == ['192.0.2.0/28', '192.0.2.16/28', '192.0.2.32/28', '192.0.2.48/28', '192.0.2.64/28', '192.0.2.80/28', '192.0.2.96/28', '192.0.2.112/28', '192.0.2.128/28', '192.0.2.144/28', '192.0.2.160/28', '192.0.2.176/28', '192.0.2.192/28', '192.0.2.208/28', '192.0.2.224/28', '192.0.2.240/28']