import re
import unittest
from oslo_config import types
def test_equal_with_same_max_and_no_min(self):
    self.assertTrue(types.Port(max=123) == types.Port(max=123))