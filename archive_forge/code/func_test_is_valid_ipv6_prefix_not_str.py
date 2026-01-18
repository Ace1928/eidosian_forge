import logging
import unittest
from os_ken.services.protocols.bgp.utils import validation
def test_is_valid_ipv6_prefix_not_str(self):
    self.assertEqual(False, validation.is_valid_ipv6_prefix(1234))