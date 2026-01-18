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
def testDefaultFields_EnumOkIfTypeKnown(self):
    """Test enum fields accept valid default values when type is known."""
    field = messages.EnumField(descriptor.FieldDescriptor.Label, 1, default='REPEATED')
    self.assertEquals(descriptor.FieldDescriptor.Label.REPEATED, field.default)