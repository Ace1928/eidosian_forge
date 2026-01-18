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
def testMessageFieldValueToMessage(self):

    class MyMessage(messages.Message):
        pass

    class HasMessage(messages.Message):
        field = messages.MessageField(MyMessage, 1)
    instance = MyMessage()
    self.assertTrue(instance is HasMessage.field.value_to_message(instance))