import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
def testEncodeCustom(self):
    decoded_message = protojson.encode_message(MyMessage(a_custom=1))
    self.CompareEncoded('{"a_custom": 1}', decoded_message)