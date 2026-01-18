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
def testNumberAttribute(self):
    """Test setting the number attribute."""

    def action(field_class):
        self.assertRaises(messages.InvalidNumberError, field_class, 0)
        self.assertRaises(messages.InvalidNumberError, field_class, -1)
        self.assertRaises(messages.InvalidNumberError, field_class, messages.MAX_FIELD_NUMBER + 1)
        self.assertRaises(messages.InvalidNumberError, field_class, messages.FIRST_RESERVED_FIELD_NUMBER)
        self.assertRaises(messages.InvalidNumberError, field_class, messages.LAST_RESERVED_FIELD_NUMBER)
        self.assertRaises(messages.InvalidNumberError, field_class, '1')
        field_class(number=1)
    self.ActionOnAllFieldClasses(action)