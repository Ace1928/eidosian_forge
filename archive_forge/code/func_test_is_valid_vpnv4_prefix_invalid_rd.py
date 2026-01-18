import logging
import unittest
from os_ken.services.protocols.bgp.utils import validation
def test_is_valid_vpnv4_prefix_invalid_rd(self):
    self.assertEqual(False, validation.is_valid_vpnv4_prefix('foo:bar:10.0.0.1/24'))