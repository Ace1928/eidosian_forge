import logging
import unittest
from os_ken.services.protocols.bgp.utils import validation
def test_is_valid_ethernet_tag_id(self):
    self.assertTrue(validation.is_valid_ethernet_tag_id(100))