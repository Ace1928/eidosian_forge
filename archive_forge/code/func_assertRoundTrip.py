import datetime
import json
import math
import unittest
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import extra_types
def assertRoundTrip(self, value):
    if isinstance(value, extra_types._JSON_PROTO_TYPES):
        self.assertEqual(value, extra_types._PythonValueToJsonProto(extra_types._JsonProtoToPythonValue(value)))
    else:
        self.assertEqual(value, extra_types._JsonProtoToPythonValue(extra_types._PythonValueToJsonProto(value)))