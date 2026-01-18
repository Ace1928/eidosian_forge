import re
import unittest
from oslo_config import types
def test_invalid_characters(self):
    self.assertInvalid('"host"')
    self.assertInvalid("h'ost'")
    self.assertInvalid("h'ost")
    self.assertInvalid('h$ost')
    self.assertInvalid('host_01.co.uk')
    self.assertInvalid('h%ost')
    self.assertInvalid('host;name=99')
    self.assertInvalid('___site0.1001')
    self.assertInvalid('_site01001')
    self.assertInvalid('host..name')
    self.assertInvalid('.host.name.com')
    self.assertInvalid('no spaces')