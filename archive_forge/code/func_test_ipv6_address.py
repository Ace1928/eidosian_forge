import re
import unittest
from oslo_config import types
def test_ipv6_address(self):
    self.assertInvalid('abcd:ef::1')