import pickle
import re
import sys
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def testUnrecognizedFieldInvalidVariant(self):

    class MyMessage(messages.Message):
        field1 = messages.IntegerField(1)
    message1 = MyMessage()
    self.assertRaises(TypeError, message1.set_unrecognized_field, 'unknown4', {'unhandled': 'type'}, None)
    self.assertRaises(TypeError, message1.set_unrecognized_field, 'unknown4', {'unhandled': 'type'}, 123)