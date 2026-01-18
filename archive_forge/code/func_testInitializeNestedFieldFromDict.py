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
def testInitializeNestedFieldFromDict(self):
    """Tests initializing nested fields from dict."""

    class SimpleMessage(messages.Message):
        required = messages.IntegerField(1, required=True)

    class NestedMessage(messages.Message):
        simple = messages.MessageField(SimpleMessage, 1)

    class RepeatedMessage(messages.Message):
        simple = messages.MessageField(SimpleMessage, 1, repeated=True)
    nested_message1 = NestedMessage(simple={'required': 10})
    self.assertTrue(nested_message1.is_initialized())
    self.assertTrue(nested_message1.simple.is_initialized())
    nested_message2 = NestedMessage()
    nested_message2.simple = {'required': 10}
    self.assertTrue(nested_message2.is_initialized())
    self.assertTrue(nested_message2.simple.is_initialized())
    repeated_values = [{}, {'required': 10}, SimpleMessage(required=20)]
    repeated_message1 = RepeatedMessage(simple=repeated_values)
    self.assertEquals(3, len(repeated_message1.simple))
    self.assertFalse(repeated_message1.is_initialized())
    repeated_message1.simple[0].required = 0
    self.assertTrue(repeated_message1.is_initialized())
    repeated_message2 = RepeatedMessage()
    repeated_message2.simple = repeated_values
    self.assertEquals(3, len(repeated_message2.simple))
    self.assertFalse(repeated_message2.is_initialized())
    repeated_message2.simple[0].required = 0
    self.assertTrue(repeated_message2.is_initialized())