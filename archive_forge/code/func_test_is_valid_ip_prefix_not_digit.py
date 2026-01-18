import logging
import unittest
from os_ken.services.protocols.bgp.utils import validation
def test_is_valid_ip_prefix_not_digit(self):
    self.assertEqual(False, validation.is_valid_ip_prefix('foo', 32))