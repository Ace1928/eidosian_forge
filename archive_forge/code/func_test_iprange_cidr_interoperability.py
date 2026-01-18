from ast import literal_eval
import pickle
import sys
import sys
import pytest
from netaddr import (
def test_iprange_cidr_interoperability():
    assert IPRange('192.0.2.5', '192.0.2.10').cidrs() == [IPNetwork('192.0.2.5/32'), IPNetwork('192.0.2.6/31'), IPNetwork('192.0.2.8/31'), IPNetwork('192.0.2.10/32')]
    assert IPRange('fe80::', 'fe80::ffff:ffff:ffff:ffff').cidrs() == [IPNetwork('fe80::/64')]