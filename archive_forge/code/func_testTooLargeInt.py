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
def testTooLargeInt(self):
    """Test that numbers too large are rejected."""
    self.assertRaises(messages.EnumDefinitionError, messages.Enum.def_enum, {'Bad': 2 ** 29}, 'BadEnum')