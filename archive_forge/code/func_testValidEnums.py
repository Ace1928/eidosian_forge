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
def testValidEnums(self):
    message_json = '{"field_one": "VALUE_ONE"}'
    message = encoding.JsonToMessage(MessageWithEnum, message_json)
    self.assertEqual(MessageWithEnum.ThisEnum.VALUE_ONE, message.field_one)
    self.assertEqual(MessageWithEnum.ThisEnum.VALUE_TWO, message.field_two)
    self.assertEqual(json.loads(message_json), json.loads(encoding.MessageToJson(message)))