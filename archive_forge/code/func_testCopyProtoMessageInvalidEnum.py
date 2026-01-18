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
def testCopyProtoMessageInvalidEnum(self):
    json_msg = '{"field_one": "BAD_VALUE"}'
    orig_msg = encoding.JsonToMessage(MessageWithEnum, json_msg)
    new_msg = encoding.CopyProtoMessage(orig_msg)
    for msg in (orig_msg, new_msg):
        self.assertEqual(msg.all_unrecognized_fields(), ['field_one'])
        self.assertEqual(msg.get_unrecognized_field_info('field_one', value_default=None), ('BAD_VALUE', messages.Variant.ENUM))