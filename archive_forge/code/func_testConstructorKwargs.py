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
def testConstructorKwargs(self):
    """Test kwargs via constructor."""

    class SomeMessage(messages.Message):
        name = messages.StringField(1)
        number = messages.IntegerField(2)
    expected = SomeMessage()
    expected.name = 'my name'
    expected.number = 200
    self.assertEquals(expected, SomeMessage(name='my name', number=200))