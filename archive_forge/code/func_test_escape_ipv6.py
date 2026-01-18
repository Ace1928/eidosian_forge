import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
def test_escape_ipv6(self):
    self.assertEqual('[1234::1234]', netutils.escape_ipv6('1234::1234'))
    self.assertEqual('127.0.0.1', netutils.escape_ipv6('127.0.0.1'))