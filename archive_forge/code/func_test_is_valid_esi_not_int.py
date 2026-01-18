import logging
import unittest
from os_ken.services.protocols.bgp.utils import validation
def test_is_valid_esi_not_int(self):
    self.assertEqual(False, validation.is_valid_esi('foo'))