import datetime
import json
import math
import unittest
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import extra_types
def testInvalidProtos(self):
    with self.assertRaises(exceptions.InvalidDataError):
        extra_types._ValidateJsonValue(extra_types.JsonValue())
    with self.assertRaises(exceptions.InvalidDataError):
        extra_types._ValidateJsonValue(extra_types.JsonValue(is_null=True, string_value='a'))
    with self.assertRaises(exceptions.InvalidDataError):
        extra_types._ValidateJsonValue(extra_types.JsonValue(integer_value=3, string_value='a'))