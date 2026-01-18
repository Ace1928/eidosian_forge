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
def testConstructorNotAField(self):
    """Test kwargs via constructor with wrong names."""

    class SomeMessage(messages.Message):
        pass
    self.assertRaisesWithRegexpMatch(AttributeError, 'May not assign arbitrary value does_not_exist to message SomeMessage', SomeMessage, does_not_exist=10)