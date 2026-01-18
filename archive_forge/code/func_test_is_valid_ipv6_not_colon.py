import logging
import unittest
from os_ken.services.protocols.bgp.utils import validation
def test_is_valid_ipv6_not_colon(self):
    self.assertEqual(False, validation.is_valid_ipv6('fe80--0011-aabb-ccdd-eeff'))