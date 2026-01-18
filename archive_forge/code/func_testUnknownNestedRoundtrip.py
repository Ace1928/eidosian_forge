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
def testUnknownNestedRoundtrip(self):
    json_message = '{"field": "abc", "submessage": {"a": 1, "b": "foo"}}'
    message = encoding.JsonToMessage(SimpleMessage, json_message)
    self.assertEqual(json.loads(json_message), json.loads(encoding.MessageToJson(message)))