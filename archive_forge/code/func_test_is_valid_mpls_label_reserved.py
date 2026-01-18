import logging
import unittest
from os_ken.services.protocols.bgp.utils import validation
def test_is_valid_mpls_label_reserved(self):
    self.assertEqual(False, validation.is_valid_mpls_label(4))