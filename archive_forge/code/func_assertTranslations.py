import datetime
import json
import math
import unittest
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import extra_types
def assertTranslations(self, py_value, json_proto):
    self.assertEqual(py_value, extra_types._JsonProtoToPythonValue(json_proto))
    self.assertEqual(json_proto, extra_types._PythonValueToJsonProto(py_value))