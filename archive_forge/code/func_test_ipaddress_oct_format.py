import pickle
import types
import random
import pytest
from netaddr import (
def test_ipaddress_oct_format():
    assert oct(IPAddress(4294967295)) == '0o37777777777'
    assert oct(IPAddress(0)) == '0o0'