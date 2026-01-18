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
def testIncludeFields(self):
    msg = SimpleMessage()
    self.assertEqual('{}', encoding.MessageToJson(msg))
    self.assertEqual('{"field": null}', encoding.MessageToJson(msg, include_fields=['field']))
    self.assertEqual('{"repfield": []}', encoding.MessageToJson(msg, include_fields=['repfield']))