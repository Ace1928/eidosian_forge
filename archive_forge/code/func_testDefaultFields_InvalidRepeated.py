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
def testDefaultFields_InvalidRepeated(self):
    """Test default field does not accept defaults."""
    self.assertRaisesWithRegexpMatch(messages.FieldDefinitionError, 'Repeated fields may not have defaults', messages.StringField, 1, repeated=True, default=[1, 2, 3])