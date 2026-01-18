import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
def testEnumerationNegativeTestWithEmptyString(self):
    """The enum value is an empty string."""
    message = protojson.decode_message(MyMessage, '{"an_enum": ""}')
    expected_message = MyMessage()
    self.assertEquals(expected_message, message)
    self.assertEquals('{"an_enum": ""}', protojson.encode_message(message))