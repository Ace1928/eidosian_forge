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
def testIsInitialized(self):
    """Tests is_initialized."""

    class SimpleMessage(messages.Message):
        required = messages.IntegerField(1, required=True)
    simple_message = SimpleMessage()
    self.assertFalse(simple_message.is_initialized())
    simple_message.required = 10
    self.assertTrue(simple_message.is_initialized())