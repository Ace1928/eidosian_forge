import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
def testDecodeEmptyMessage(self):
    self.assertEqual(MyMessage(a_string='{decoded}'), self.protojson.decode_message(MyMessage, '{"a_string": ""}'))