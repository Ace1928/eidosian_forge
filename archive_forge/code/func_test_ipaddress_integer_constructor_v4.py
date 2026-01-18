import pickle
import types
import random
import pytest
from netaddr import (
def test_ipaddress_integer_constructor_v4():
    assert IPAddress(1) == IPAddress('0.0.0.1')
    assert IPAddress(1, 4) == IPAddress('0.0.0.1')
    assert IPAddress(1, 6) == IPAddress('::1')
    assert IPAddress(10) == IPAddress('0.0.0.10')