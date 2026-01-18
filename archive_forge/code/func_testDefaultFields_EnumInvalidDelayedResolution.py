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
def testDefaultFields_EnumInvalidDelayedResolution(self):
    """Test that enum fields raise errors upon delayed resolution error."""
    field = messages.EnumField('apitools.base.protorpclite.descriptor.FieldDescriptor.Label', 1, default=200)
    self.assertRaisesWithRegexpMatch(TypeError, 'No such value for 200 in Enum Label', getattr, field, 'default')