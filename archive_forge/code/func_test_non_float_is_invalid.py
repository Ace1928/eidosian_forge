import re
import unittest
from oslo_config import types
def test_non_float_is_invalid(self):
    self.assertInvalid('123,345')
    self.assertInvalid('foo')