import re
import unittest
from oslo_config import types
def test_dict_of_dicts(self):
    self.type_instance = types.Dict(types.Dict(types.String(), bounds=True))
    self.assertConvertedValue('k1:{k1:v1,k2:v2},k2:{k3:v3}', {'k1': {'k1': 'v1', 'k2': 'v2'}, 'k2': {'k3': 'v3'}})