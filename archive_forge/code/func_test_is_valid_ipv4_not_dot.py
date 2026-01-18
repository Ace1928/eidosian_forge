import logging
import unittest
from os_ken.services.protocols.bgp.utils import validation
def test_is_valid_ipv4_not_dot(self):
    self.assertEqual(False, validation.is_valid_ipv4('192:168:0:1'))