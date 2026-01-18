import re
import unittest
from oslo_config import types
def test_equal_with_equal_custom_item_types(self):
    it1 = types.Integer()
    it2 = types.Integer()
    self.assertTrue(types.Dict(it1) == types.Dict(it2))