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
def testDefaultFields_Single(self):
    """Test default field is correct type (single)."""
    defaults = {messages.IntegerField: 10, messages.FloatField: 1.5, messages.BooleanField: False, messages.BytesField: b'abc', messages.StringField: u'abc'}

    def action(field_class):
        field_class(1, default=defaults[field_class])
    self.ActionOnAllFieldClasses(action)
    defaults[messages.StringField] = 'abc'
    self.ActionOnAllFieldClasses(action)