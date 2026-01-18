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
def testIgnoredEnumsWithDefaults(self):
    json_with_typo = '{"field_two": "VALUE_OEN"}'
    message = encoding.JsonToMessage(MessageWithEnum, json_with_typo)
    self.assertEqual(MessageWithEnum.ThisEnum.VALUE_TWO, message.field_two)
    self.assertEqual(json.loads(json_with_typo), json.loads(encoding.MessageToJson(message)))