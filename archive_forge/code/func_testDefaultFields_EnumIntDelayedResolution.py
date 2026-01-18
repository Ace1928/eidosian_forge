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
def testDefaultFields_EnumIntDelayedResolution(self):
    """Test that enum fields resolve default integers."""
    field = messages.EnumField('apitools.base.protorpclite.descriptor.FieldDescriptor.Label', 1, default=2)
    self.assertEquals(descriptor.FieldDescriptor.Label.REQUIRED, field.default)