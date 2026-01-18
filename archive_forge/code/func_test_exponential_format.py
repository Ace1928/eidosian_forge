import re
import unittest
from oslo_config import types
def test_exponential_format(self):
    v = self.type_instance('123e-2')
    self.assertAlmostEqual(v, 1.23)