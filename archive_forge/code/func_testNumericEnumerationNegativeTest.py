import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
def testNumericEnumerationNegativeTest(self):
    """Test with an invalid number for the enum value."""
    message = protojson.decode_message(MyMessage, '{"an_enum": 89}')
    expected_message = MyMessage()
    self.assertEquals(expected_message, message)
    self.assertEquals('{"an_enum": 89}', protojson.encode_message(message))