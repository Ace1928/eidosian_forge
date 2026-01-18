import re
import unittest
from oslo_config import types
def test_invalid_host_addresses(self):
    self.assertInvalid('-1')
    self.assertInvalid('3.14')
    self.assertInvalid('10.0')
    self.assertInvalid('host..name')
    self.assertInvalid('org.10')
    self.assertInvalid('0.0.00')