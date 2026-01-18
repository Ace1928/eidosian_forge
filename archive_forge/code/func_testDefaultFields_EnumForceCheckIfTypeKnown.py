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
def testDefaultFields_EnumForceCheckIfTypeKnown(self):
    """Test that enum fields validate default values if type is known."""
    self.assertRaisesWithRegexpMatch(TypeError, 'No such value for NOT_A_LABEL in Enum Label', messages.EnumField, descriptor.FieldDescriptor.Label, 1, default='NOT_A_LABEL')