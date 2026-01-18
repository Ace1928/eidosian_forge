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
def testInvalidVariant(self):
    """Test field with invalid variants."""

    def action(field_class):
        if field_class is not message_types.DateTimeField:
            self.assertRaises(messages.InvalidVariantError, field_class, 1, variant=messages.Variant.ENUM)
    self.ActionOnAllFieldClasses(action)