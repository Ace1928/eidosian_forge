from ast import literal_eval
import pickle
import sys
import sys
import pytest
from netaddr import (
def test_iprange_membership():
    assert IPRange('192.0.2.5', '192.0.2.10') in IPRange('192.0.2.1', '192.0.2.254')
    assert IPRange('fe80::1', 'fe80::fffe') in IPRange('fe80::', 'fe80::ffff:ffff:ffff:ffff')
    assert IPRange('192.0.2.5', '192.0.2.10') not in IPRange('::', '::255.255.255.255')
    net = IPNetwork('10.0.0.0/30')
    assert net in IPRange(net.first, net.last)