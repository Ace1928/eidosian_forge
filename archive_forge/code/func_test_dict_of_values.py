import re
import unittest
from oslo_config import types
def test_dict_of_values(self):
    self.assertConvertedValue(' foo: bar, baz: 123 ', {'foo': 'bar', 'baz': '123'})