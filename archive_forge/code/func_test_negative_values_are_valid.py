import re
import unittest
from oslo_config import types
def test_negative_values_are_valid(self):
    self.assertConvertedValue('-123', -123)