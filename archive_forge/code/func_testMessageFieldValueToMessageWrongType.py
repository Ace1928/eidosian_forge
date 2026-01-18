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
def testMessageFieldValueToMessageWrongType(self):

    class MyMessage(messages.Message):
        pass

    class MyOtherMessage(messages.Message):
        pass

    class HasMessage(messages.Message):
        field = messages.MessageField(MyMessage, 1)
    instance = MyOtherMessage()
    self.assertRaisesWithRegexpMatch(messages.EncodeError, 'Expected type MyMessage, got MyOtherMessage: <MyOtherMessage>', HasMessage.field.value_to_message, instance)