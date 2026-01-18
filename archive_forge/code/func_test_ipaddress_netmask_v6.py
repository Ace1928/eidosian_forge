import pickle
import pytest
from netaddr import IPAddress, IPNetwork
def test_ipaddress_netmask_v6():
    assert IPAddress('::').netmask_bits() == 0
    assert IPAddress('8000::').netmask_bits() == 1
    assert IPAddress('ffff:ffff:ffff:ffff::').netmask_bits() == 64
    assert IPAddress('ffff:ffff:ffff:ffff:ffff:ffff:ffff::').netmask_bits() == 112
    assert IPAddress('ffff:ffff:ffff:ffff:ffff:ffff:ffff:fffe').netmask_bits() == 127
    assert IPAddress('ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff').netmask_bits() == 128
    assert IPAddress('fe80::1').netmask_bits() == 128