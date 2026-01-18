import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
def testDecodeRepeatedCustom(self):
    message = protojson.decode_message(MyMessage, '{"a_repeated_custom": [1, 2, 3]}')
    self.assertEquals(MyMessage(a_repeated_custom=[1, 2, 3]), message)