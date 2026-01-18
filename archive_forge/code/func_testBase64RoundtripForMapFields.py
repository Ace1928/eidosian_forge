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
def testBase64RoundtripForMapFields(self):
    raw_data = b'\xff\x0f\x80'
    encoded_data = '/w+A'
    safe_encoded_data = base64.urlsafe_b64encode(raw_data).decode('utf-8')
    self.assertEqual(raw_data, base64.b64decode(encoded_data))
    json_data = '{"1st": "%s"}' % encoded_data
    msg = encoding.JsonToMessage(MapToBytesValue, json_data)
    self.assertEqual(raw_data, msg.additionalProperties[0].value)
    from_msg_json_data = encoding.MessageToJson(msg)
    self.assertEqual(safe_encoded_data, json.loads(from_msg_json_data)['1st'])
    redone_msg = encoding.JsonToMessage(MapToBytesValue, from_msg_json_data)
    self.assertEqual(raw_data, redone_msg.additionalProperties[0].value)