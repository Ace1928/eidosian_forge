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
def testIgnoredEnums(self):
    json_with_typo = '{"field_one": "VALUE_OEN"}'
    message = encoding.JsonToMessage(MessageWithEnum, json_with_typo)
    self.assertEqual(None, message.field_one)
    self.assertEqual(('VALUE_OEN', messages.Variant.ENUM), message.get_unrecognized_field_info('field_one'))
    self.assertEqual(json.loads(json_with_typo), json.loads(encoding.MessageToJson(message)))
    empty_json = ''
    message = encoding.JsonToMessage(MessageWithEnum, empty_json)
    self.assertEqual(None, message.field_one)