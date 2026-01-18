import re
import unittest
from oslo_config import types
def test_equal_with_same_min_and_no_max(self):
    self.assertTrue(types.Port(min=123) == types.Port(min=123))