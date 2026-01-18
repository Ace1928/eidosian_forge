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
def testValidateCastingElement(self):
    field = messages.FloatField(1)
    self.assertEquals(type(field.validate_element(12)), float)
    self.assertEquals(type(field.validate_element(12.0)), float)
    field = messages.IntegerField(1)
    self.assertEquals(type(field.validate_element(12)), int)
    self.assertRaises(messages.ValidationError, field.validate_element, 12.0)