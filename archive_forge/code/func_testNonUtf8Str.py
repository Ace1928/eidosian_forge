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
def testNonUtf8Str(self):
    """Test validation fails for non-UTF-8 StringField values."""

    class Thing(messages.Message):
        string_field = messages.StringField(2)
    thing = Thing()
    self.assertRaisesWithRegexpMatch(messages.ValidationError, 'Field string_field encountered non-UTF-8 string', setattr, thing, 'string_field', test_util.BINARY)