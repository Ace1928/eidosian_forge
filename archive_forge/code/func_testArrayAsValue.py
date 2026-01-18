import datetime
import json
import math
import unittest
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import extra_types
def testArrayAsValue(self):
    array_json = '[3, "four", false]'
    array = [3, 'four', False]
    value = encoding.JsonToMessage(extra_types.JsonValue, array_json)
    self.assertTrue(isinstance(value, extra_types.JsonValue))
    self.assertEqual(array, encoding.MessageToPyValue(value))