import re
import unittest
from oslo_config import types
def test_equal_with_same_regex(self):
    t1 = types.String(regex=re.compile('^[A-Z]'))
    t2 = types.String(regex=re.compile('^[A-Z]'))
    self.assertTrue(t1 == t2)