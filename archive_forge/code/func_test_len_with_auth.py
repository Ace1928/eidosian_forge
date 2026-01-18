import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib.packet import ipv6
def test_len_with_auth(self):
    self.setUp_with_auth()
    self.assertEqual(len(self.ip), 40 + len(self.auth))