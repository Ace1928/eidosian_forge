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
def testCompareAutoInitializedRepeatedFields(self):

    class SomeMessage(messages.Message):
        repeated = messages.IntegerField(1, repeated=True)
    message1 = SomeMessage(repeated=[])
    message2 = SomeMessage()
    self.assertEquals(message1, message2)