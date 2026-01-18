import logging
import unittest
from os_ken.services.protocols.bgp.utils import validation
def test_is_valid_med_negative(self):
    self.assertEqual(False, validation.is_valid_med(-1))