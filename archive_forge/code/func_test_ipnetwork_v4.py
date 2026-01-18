import pickle
import types
import random
import pytest
from netaddr import (
@pytest.mark.parametrize(('value', 'ipaddr', 'network', 'cidr', 'broadcast', 'netmask', 'hostmask', 'size'), [('192.0.2.1', IPAddress('192.0.2.1'), IPAddress('192.0.2.1'), IPNetwork('192.0.2.1/32'), None, IPAddress('255.255.255.255'), IPAddress('0.0.0.0'), 1), ('192.0.2.0/24', IPAddress('192.0.2.0'), IPAddress('192.0.2.0'), IPNetwork('192.0.2.0/24'), IPAddress('192.0.2.255'), IPAddress('255.255.255.0'), IPAddress('0.0.0.255'), 256), ('192.0.3.112/22', IPAddress('192.0.3.112'), IPAddress('192.0.0.0'), IPNetwork('192.0.0.0/22'), IPAddress('192.0.3.255'), IPAddress('255.255.252.0'), IPAddress('0.0.3.255'), 1024)])
def test_ipnetwork_v4(value, ipaddr, network, cidr, broadcast, netmask, hostmask, size):
    net = IPNetwork(value)
    assert net.ip == ipaddr
    assert net.network == network
    assert net.cidr == cidr
    assert net.broadcast == broadcast
    assert net.netmask == netmask
    assert net.hostmask == hostmask
    assert net.size == size