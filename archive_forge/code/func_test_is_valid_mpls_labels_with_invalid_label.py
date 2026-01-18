import logging
import unittest
from os_ken.services.protocols.bgp.utils import validation
def test_is_valid_mpls_labels_with_invalid_label(self):
    self.assertEqual(False, validation.is_valid_mpls_labels(['foo', 200]))