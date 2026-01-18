import re
import unittest
from oslo_config import types
def test_equal_with_same_min_and_max(self):
    t1 = types.Port(min=1, max=123)
    t2 = types.Port(min=1, max=123)
    self.assertTrue(t1 == t2)