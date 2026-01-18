import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
def testDecodeInvalidDateTime(self):
    self.assertRaises(messages.DecodeError, protojson.decode_message, MyMessage, '{"a_datetime": "invalid"}')