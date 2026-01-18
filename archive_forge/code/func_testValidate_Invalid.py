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
def testValidate_Invalid(self):
    """Test validation of valid values."""
    values = {messages.IntegerField: '10', messages.FloatField: 'blah', messages.BooleanField: 0, messages.BytesField: 10.2, messages.StringField: 42}

    def action(field_class):
        field = field_class(1)
        self.assertRaises(messages.ValidationError, field.validate, values[field_class])
        field = field_class(1, required=True)
        self.assertRaises(messages.ValidationError, field.validate, values[field_class])
        field = field_class(1, repeated=True)
        self.assertRaises(messages.ValidationError, field.validate, [values[field_class]])
        self.assertRaises(messages.ValidationError, field.validate, (values[field_class],))
    self.ActionOnAllFieldClasses(action)