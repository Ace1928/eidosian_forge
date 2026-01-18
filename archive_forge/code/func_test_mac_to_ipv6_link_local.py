import pickle
import random
from netaddr import (
def test_mac_to_ipv6_link_local():
    mac = EUI('00-0F-1F-12-E7-33')
    ip = mac.ipv6_link_local()
    assert ip == IPAddress('fe80::20f:1fff:fe12:e733')
    assert ip.is_link_local()
    assert mac.eui64() == EUI('00-0F-1F-FF-FE-12-E7-33')