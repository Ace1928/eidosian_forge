import datetime
import json
import math
import unittest
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import extra_types
def testObjectAsValue(self):
    obj_json = '{"works": true}'
    obj = {'works': True}
    value = encoding.JsonToMessage(extra_types.JsonValue, obj_json)
    self.assertTrue(isinstance(value, extra_types.JsonValue))
    self.assertEqual(obj, encoding.MessageToPyValue(value))