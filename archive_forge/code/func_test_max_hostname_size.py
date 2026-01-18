import re
import unittest
from oslo_config import types
def test_max_hostname_size(self):
    test_str = '.'.join(('x' * 31 for x in range(8)))
    self.assertEqual(255, len(test_str))
    self.assertInvalid(test_str)
    self.assertConvertedEqual(test_str[:-2])