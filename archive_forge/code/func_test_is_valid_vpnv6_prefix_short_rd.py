import logging
import unittest
from os_ken.services.protocols.bgp.utils import validation
def test_is_valid_vpnv6_prefix_short_rd(self):
    self.assertEqual(False, validation.is_valid_vpnv6_prefix('100:eeff/64'))