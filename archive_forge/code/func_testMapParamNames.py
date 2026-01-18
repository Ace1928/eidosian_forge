import unittest
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import util
def testMapParamNames(self):
    params = ['path_field', 'enum_field']
    remapped_params = ['str_field', 'enum_field']
    self.assertEqual(remapped_params, util.MapParamNames(params, MessageWithRemappings))