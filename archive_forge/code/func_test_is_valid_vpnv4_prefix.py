import logging
import unittest
from os_ken.services.protocols.bgp.utils import validation
def test_is_valid_vpnv4_prefix(self):
    self.assertTrue(validation.is_valid_vpnv4_prefix('100:200:10.0.0.1/24'))