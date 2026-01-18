import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
def testEncode(self):
    self.assertEqual('{"a_string": "{encoded}xyz"}', self.protojson.encode_message(MyMessage(a_string='xyz')))