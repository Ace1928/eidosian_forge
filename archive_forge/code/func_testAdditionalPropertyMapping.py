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
def testAdditionalPropertyMapping(self):
    msg = AdditionalPropertiesMessage()
    msg.additionalProperties = [AdditionalPropertiesMessage.AdditionalProperty(key='key_one', value='value_one'), AdditionalPropertiesMessage.AdditionalProperty(key=u'key_twð', value='value_two')]
    encoded_msg = encoding.MessageToJson(msg)
    self.assertEqual({'key_one': 'value_one', u'key_twð': 'value_two'}, json.loads(encoded_msg))
    new_msg = encoding.JsonToMessage(type(msg), encoded_msg)
    self.assertEqual(set(('key_one', u'key_twð')), set([x.key for x in new_msg.additionalProperties]))
    self.assertIsNot(msg, new_msg)
    new_msg.additionalProperties.pop()
    self.assertEqual(1, len(new_msg.additionalProperties))
    self.assertEqual(2, len(msg.additionalProperties))