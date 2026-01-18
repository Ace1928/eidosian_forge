import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
def testAlphaEnumeration(self):
    """Test that alpha enum values work."""
    message = protojson.decode_message(MyMessage, '{"an_enum": "RED"}')
    expected_message = MyMessage()
    expected_message.an_enum = MyMessage.Color.RED
    self.assertEquals(expected_message, message)