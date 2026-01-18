import re
import unittest
from oslo_config import types
def test_list_of_custom_type(self):
    self.type_instance = types.List(types.Integer())
    self.assertConvertedValue('1,2,3,5', [1, 2, 3, 5])