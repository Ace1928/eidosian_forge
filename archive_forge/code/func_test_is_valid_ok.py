import unittest
import logging
import struct
import inspect
from os_ken.ofproto import inet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import ipv6
from os_ken.lib.packet import packet
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import vrrp
from os_ken.lib import addrconv
def test_is_valid_ok(self):
    self.assertTrue(self._test_is_valid())