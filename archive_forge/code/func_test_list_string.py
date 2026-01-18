import re
import unittest
from oslo_config import types
def test_list_string(self):
    t = types.List(item_type=types.String())
    test_list = ['foo', Exception(' bar ')]
    self.assertEqual(['foo," bar "'], t.format_defaults('', sample_default=test_list))