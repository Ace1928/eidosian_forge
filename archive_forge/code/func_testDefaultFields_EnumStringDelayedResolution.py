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
def testDefaultFields_EnumStringDelayedResolution(self):
    """Test that enum fields resolve default strings."""
    field = messages.EnumField('apitools.base.protorpclite.descriptor.FieldDescriptor.Label', 1, default='OPTIONAL')
    self.assertEquals(descriptor.FieldDescriptor.Label.OPTIONAL, field.default)