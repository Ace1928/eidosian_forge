import logging
import unittest
from os_ken.services.protocols.bgp.utils import validation
def test_is_valid_mac_hyphenation(self):
    self.assertTrue(validation.is_valid_mac('aa-bb-cc-dd-ee-ff'))