import logging
import unittest
from os_ken.services.protocols.bgp.utils import validation
def test_is_valid_vpnv4_prefix_not_str(self):
    self.assertEqual(False, validation.is_valid_vpnv4_prefix(1234))