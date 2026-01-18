import datetime
import json
import math
import unittest
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import extra_types
def testJsonNumberEncoding(self):
    seventeen = extra_types.JsonValue(integer_value=17)
    self.assertRoundTrip(17)
    self.assertRoundTrip(seventeen)
    self.assertTranslations(17, seventeen)
    json_pi = extra_types.JsonValue(double_value=math.pi)
    self.assertRoundTrip(math.pi)
    self.assertRoundTrip(json_pi)
    self.assertTranslations(math.pi, json_pi)