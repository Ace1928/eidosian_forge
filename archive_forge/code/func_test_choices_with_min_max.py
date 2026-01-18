import re
import unittest
from oslo_config import types
def test_choices_with_min_max(self):
    self.assertRaises(ValueError, types.Port, min=100, choices=[50, 60])
    self.assertRaises(ValueError, types.Port, max=10, choices=[50, 60])
    types.Port(min=10, max=100, choices=[50, 60])