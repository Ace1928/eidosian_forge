import re
import unittest
from oslo_config import types
def test_ignore_case_raises(self):
    self.type_instance = types.String(choices=['foo', 'bar'], ignore_case=False)
    self.assertRaises(ValueError, self.assertConvertedValue, 'Foo', 'Foo')