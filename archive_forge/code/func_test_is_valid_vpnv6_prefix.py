import logging
import unittest
from os_ken.services.protocols.bgp.utils import validation
def test_is_valid_vpnv6_prefix(self):
    self.assertTrue(validation.is_valid_vpnv6_prefix('100:200:fe80::0011:aabb:ccdd:eeff/64'))