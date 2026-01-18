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
def testValidateElement(self):
    """Test validation of valid values."""
    values = {messages.IntegerField: (10, -1, 0), messages.FloatField: (1.5, -1.5, 3), messages.BooleanField: (True, False), messages.BytesField: (b'abc',), messages.StringField: (u'abc',)}

    def action(field_class):
        field = field_class(1)
        for value in values[field_class]:
            field.validate_element(value)
        field = field_class(1, required=True)
        for value in values[field_class]:
            field.validate_element(value)
        field = field_class(1, repeated=True)
        self.assertRaises(messages.ValidationError, field.validate_element, [])
        self.assertRaises(messages.ValidationError, field.validate_element, ())
        for value in values[field_class]:
            field.validate_element(value)
        self.assertRaises(messages.ValidationError, field.validate_element, list(values[field_class]))
        self.assertRaises(messages.ValidationError, field.validate_element, values[field_class])
    self.ActionOnAllFieldClasses(action)