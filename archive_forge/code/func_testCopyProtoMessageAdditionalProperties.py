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
def testCopyProtoMessageAdditionalProperties(self):
    msg = AdditionalPropertiesMessage(additionalProperties=[AdditionalPropertiesMessage.AdditionalProperty(key='key', value='value')])
    new_msg = encoding.CopyProtoMessage(msg)
    self.assertEqual(len(new_msg.additionalProperties), 1)
    self.assertEqual(new_msg.additionalProperties[0].key, 'key')
    self.assertEqual(new_msg.additionalProperties[0].value, 'value')