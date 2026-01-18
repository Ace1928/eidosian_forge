import datetime
import json
import math
import unittest
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import extra_types
def testJsonObjectPropertyTranslation(self):
    value = extra_types.JsonValue(string_value='abc')
    obj = extra_types.JsonObject(properties=[extra_types.JsonObject.Property(key='attr_name', value=value)])
    json_value = '"abc"'
    json_obj = '{"attr_name": "abc"}'
    self.assertRoundTrip(value)
    self.assertRoundTrip(obj)
    self.assertRoundTrip(json_value)
    self.assertRoundTrip(json_obj)
    self.assertEqual(json_value, encoding.MessageToJson(value))
    self.assertEqual(json_obj, encoding.MessageToJson(obj))