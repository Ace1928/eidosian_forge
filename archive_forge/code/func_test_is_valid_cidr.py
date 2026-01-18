import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
def test_is_valid_cidr(self):
    self.assertTrue(netutils.is_valid_cidr('10.0.0.0/24'))
    self.assertTrue(netutils.is_valid_cidr('10.0.0.1/32'))
    self.assertTrue(netutils.is_valid_cidr('0.0.0.0/0'))
    self.assertTrue(netutils.is_valid_cidr('2600::/64'))
    self.assertTrue(netutils.is_valid_cidr('0000:0000:0000:0000:0000:0000:0000:0001/32'))
    self.assertFalse(netutils.is_valid_cidr('10.0.0.1'))
    self.assertFalse(netutils.is_valid_cidr('10.0.0.1/33'))
    self.assertFalse(netutils.is_valid_cidr(10))