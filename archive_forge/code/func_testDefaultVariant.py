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
def testDefaultVariant(self):
    """Test that default variant is used when not set."""

    def action(field_class):
        field = field_class(1)
        self.assertEquals(field_class.DEFAULT_VARIANT, field.variant)
    self.ActionOnAllFieldClasses(action)