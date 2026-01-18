import datetime
import json
import math
import unittest
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import extra_types
def testInt64(self):

    class DogeMsg(messages.Message):
        such_string = messages.StringField(1)
        wow = messages.IntegerField(2, variant=messages.Variant.INT64)
        very_unsigned = messages.IntegerField(3, variant=messages.Variant.UINT64)
        much_repeated = messages.IntegerField(4, variant=messages.Variant.INT64, repeated=True)

    def MtoJ(msg):
        return encoding.MessageToJson(msg)

    def JtoM(class_type, json_str):
        return encoding.JsonToMessage(class_type, json_str)

    def DoRoundtrip(class_type, json_msg=None, message=None, times=4):
        if json_msg:
            json_msg = MtoJ(JtoM(class_type, json_msg))
        if message:
            message = JtoM(class_type, MtoJ(message))
        if times == 0:
            result = json_msg if json_msg else message
            return result
        return DoRoundtrip(class_type=class_type, json_msg=json_msg, message=message, times=times - 1)
    json_msg = '{"such_string": "poot", "wow": "-1234", "very_unsigned": "999", "much_repeated": ["123", "456"]}'
    out_json = MtoJ(JtoM(DogeMsg, json_msg))
    self.assertEqual(json.loads(out_json)['wow'], '-1234')
    msg = DogeMsg(such_string='wow', wow=-1234, very_unsigned=800, much_repeated=[123, 456])
    self.assertEqual(msg, DoRoundtrip(DogeMsg, message=msg))