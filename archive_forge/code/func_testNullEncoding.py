import datetime
import json
import math
import unittest
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import extra_types
def testNullEncoding(self):
    self.assertTranslations(None, extra_types.JsonValue(is_null=True))