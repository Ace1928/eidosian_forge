import pickle
import types
import random
import pytest
from netaddr import (
def test_ipaddress_inet_aton_constructor_v4():
    assert IPAddress('0x7f.0x1', flags=INET_ATON) == IPAddress('127.0.0.1')
    assert IPAddress('0x7f.0x0.0x0.0x1', flags=INET_ATON) == IPAddress('127.0.0.1')
    assert IPAddress('0177.01', flags=INET_ATON) == IPAddress('127.0.0.1')
    assert IPAddress('0x7f.0.01', flags=INET_ATON) == IPAddress('127.0.0.1')
    assert IPAddress('127', flags=INET_ATON) == IPAddress('0.0.0.127')
    assert IPAddress('127', flags=INET_ATON) == IPAddress('0.0.0.127')
    assert IPAddress('127.1', flags=INET_ATON) == IPAddress('127.0.0.1')
    assert IPAddress('127.0.1', flags=INET_ATON) == IPAddress('127.0.0.1')