import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
def testDecodeInvalidCustom(self):
    self.assertRaises(messages.ValidationError, protojson.decode_message, MyMessage, '{"a_custom": "invalid"}')