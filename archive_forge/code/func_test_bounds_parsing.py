import re
import unittest
from oslo_config import types
def test_bounds_parsing(self):
    self.type_instance = types.Dict(types.String(), bounds=True)
    self.assertConvertedValue('{foo:bar,baz:123}', {'foo': 'bar', 'baz': '123'})