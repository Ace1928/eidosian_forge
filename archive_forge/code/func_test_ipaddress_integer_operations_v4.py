import pickle
import types
import random
import pytest
from netaddr import (
def test_ipaddress_integer_operations_v4():
    assert IPAddress('192.0.2.0') + 1 == IPAddress('192.0.2.1')
    assert 1 + IPAddress('192.0.2.0') == IPAddress('192.0.2.1')
    assert IPAddress('192.0.2.1') - 1 == IPAddress('192.0.2.0')
    assert IPAddress('192.0.0.0') + IPAddress('0.0.0.42') == IPAddress('192.0.0.42')
    assert IPAddress('192.0.0.42') - IPAddress('0.0.0.42') == IPAddress('192.0.0.0')
    with pytest.raises(IndexError):
        1 - IPAddress('192.0.2.1')
    ip = IPAddress('10.0.0.1')
    ip += 1
    assert ip == IPAddress('10.0.0.2')
    ip -= 1
    assert ip == IPAddress('10.0.0.1')
    ip += IPAddress('0.0.0.42')
    assert ip == IPAddress('10.0.0.43')
    ip -= IPAddress('0.0.0.43')
    assert ip == IPAddress('10.0.0.0')
    ip = IPAddress('0.0.0.0')
    with pytest.raises(IndexError):
        ip += -1
    ip = IPAddress('255.255.255.255')
    with pytest.raises(IndexError):
        ip -= -1