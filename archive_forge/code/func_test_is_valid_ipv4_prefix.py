import logging
import unittest
from os_ken.services.protocols.bgp.utils import validation
def test_is_valid_ipv4_prefix(self):
    self.assertTrue(validation.is_valid_ipv4_prefix('10.0.0.1/24'))