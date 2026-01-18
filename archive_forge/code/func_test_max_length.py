import re
import unittest
from oslo_config import types
def test_max_length(self):
    self.type_instance = types.String(max_length=30)
    self.assertInvalid('http://www.example.com/versions')
    self.assertConvertedValue('http://www.example.com', 'http://www.example.com')