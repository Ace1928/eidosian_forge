import re
import unittest
from oslo_config import types
def assertRange(self, s, r1, r2, step=1):
    self.assertEqual(list(range(r1, r2, step)), list(self.type_instance(s)))