import base64
import datetime
import json
import sys
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import extra_types
def testEnumRemapping(self):
    msg = MessageWithRemappings(enum_field=MessageWithRemappings.SomeEnum.enum_value)
    json_message = encoding.MessageToJson(msg)
    self.assertEqual('{"enum_field": "wire_name"}', json_message)
    self.assertEqual(msg, encoding.JsonToMessage(MessageWithRemappings, json_message))