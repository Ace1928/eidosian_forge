import re
import unittest
from oslo_config import types
def test_illegal_min(self):
    self.assertRaises(ValueError, types.Port, min=-1, max=50)
    self.assertRaises(ValueError, types.Port, min=-50)