import re
import unittest
from oslo_config import types
def test_list_of_values_containing_trailing_comma(self):
    self.assertConvertedValue('foo, bar, baz, ', ['foo', 'bar', 'baz'])