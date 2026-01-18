import re
import unittest
from oslo_config import types
def assertConvertedEqual(self, value):
    self.assertConvertedValue(value, value)