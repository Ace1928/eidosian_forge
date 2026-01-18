import pickle
import types
import random
import pytest
from netaddr import (
def test_ipaddress_hex_format():
    assert hex(IPAddress(0)) == '0x0'
    assert hex(IPAddress(4294967295)) == '0xffffffff'