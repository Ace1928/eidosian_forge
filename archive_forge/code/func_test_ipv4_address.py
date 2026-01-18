import re
import unittest
from oslo_config import types
def test_ipv4_address(self):
    self.assertInvalid('192.168.0.1')