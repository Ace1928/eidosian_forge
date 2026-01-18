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
def testIsInitializedNestedField(self):
    """Tests is_initialized for nested fields."""

    class SimpleMessage(messages.Message):
        required = messages.IntegerField(1, required=True)

    class NestedMessage(messages.Message):
        simple = messages.MessageField(SimpleMessage, 1)
    simple_message = SimpleMessage()
    self.assertFalse(simple_message.is_initialized())
    nested_message = NestedMessage(simple=simple_message)
    self.assertFalse(nested_message.is_initialized())
    simple_message.required = 10
    self.assertTrue(simple_message.is_initialized())
    self.assertTrue(nested_message.is_initialized())