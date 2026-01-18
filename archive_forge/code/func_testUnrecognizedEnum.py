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
def testUnrecognizedEnum(self):
    json_msg = '{"input": "VALUE_ONE"}'
    result = encoding.JsonToMessage(UnrecognizedEnumMessage, json_msg)
    self.assertEqual(1, len(result.additionalProperties))
    self.assertEqual(UnrecognizedEnumMessage.ThisEnum.VALUE_ONE, result.additionalProperties[0].value)