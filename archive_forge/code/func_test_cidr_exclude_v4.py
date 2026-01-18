import random
from netaddr import (
def test_cidr_exclude_v4():
    assert cidr_exclude('192.0.2.1/32', '192.0.2.1/32') == []
    assert cidr_exclude('192.0.2.0/31', '192.0.2.1/32') == [IPNetwork('192.0.2.0/32')]
    assert cidr_exclude('192.0.2.0/24', '192.0.2.128/25') == [IPNetwork('192.0.2.0/25')]
    assert cidr_exclude('192.0.2.0/24', '192.0.2.128/27') == [IPNetwork('192.0.2.0/25'), IPNetwork('192.0.2.160/27'), IPNetwork('192.0.2.192/26')]
    assert cidr_exclude('192.0.2.1/32', '192.0.2.0/24') == []
    assert cidr_exclude('192.0.2.0/28', '192.0.2.16/32') == [IPNetwork('192.0.2.0/28')]
    assert cidr_exclude('192.0.1.255/32', '192.0.2.0/28') == [IPNetwork('192.0.1.255/32')]